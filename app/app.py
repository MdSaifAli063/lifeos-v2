import os
import sys
import json
import csv
from datetime import datetime
from collections import defaultdict

# Ensure sibling packages are importable when running this file directly.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, render_template_string, send_file

# Add CORS support
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

# Lazy import for transformers to avoid torch/sympy conflicts
# The app will work in "rule-based mode" if model fails to load
_transformers = None
AutoTokenizer = None
AutoModelForSeq2SeqLM = None

def _lazy_import_transformers():
    global _transformers, AutoTokenizer, AutoModelForSeq2SeqLM
    if _transformers is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            _transformers = "loaded"
        except Exception as e:
            _transformers = f"error: {e}"
            AutoTokenizer = None
            AutoModelForSeq2SeqLM = None

app = Flask(__name__)

# Enable CORS for frontend communication
if CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------------
# Episode Storage for Dashboard
# ----------------------------------
_episode_history = []

# ----------------------------------
# Model
# ----------------------------------

MODEL_NAME = os.getenv("LIFEOS_MODEL_NAME", "google/flan-t5-base")
_tokenizer = None
_model = None
_model_load_error = None
_history = []
_reward_cache = None


def _load_reward_artifacts():
    """Load reward logs used for judging evidence."""
    global _reward_cache
    if _reward_cache is not None:
        return _reward_cache

    outputs_dir = os.path.join(PROJECT_ROOT, "training_outputs")
    csv_path = os.path.join(outputs_dir, "reward_log.csv")
    json_path = os.path.join(outputs_dir, "reward_log.json")
    rows = []

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        rows.append(item)
        except Exception:
            rows = []

    if not rows and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
        except Exception:
            rows = []

    _reward_cache = rows
    return rows


def _reward_summary(rows):
    if not rows:
        return {
            "available": False,
            "message": "No reward logs found in training_outputs/"
        }

    rewards = []
    for row in rows:
        try:
            rewards.append(float(row.get("reward", 0)))
        except Exception:
            continue

    if not rewards:
        return {
            "available": False,
            "message": "Reward logs found, but reward values are invalid"
        }

    first = rewards[0]
    last = rewards[-1]
    improvement = last - first
    positive = sum(1 for r in rewards if r >= 0)
    success_rate = round((positive / len(rewards)) * 100, 2)

    return {
        "available": True,
        "points": len(rewards),
        "first_reward": round(first, 4),
        "last_reward": round(last, 4),
        "max_reward": round(max(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "avg_reward": round(sum(rewards) / len(rewards), 4),
        "reward_improvement": round(improvement, 4),
        "success_probability_pct": success_rate
    }


def get_model():
    global _tokenizer, _model, _model_load_error

    if _tokenizer is None or _model is None:
        # Lazy import to avoid torch/sympy conflicts at startup
        _lazy_import_transformers()
        
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            _model_load_error = "transformers not available (torch/sympy conflict)"
            return None, None
            
        try:
            print(f"Loading model: {MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _model_load_error = None
            print("Model Ready.")
        except Exception as exc:
            _model_load_error = str(exc)
            print(f"Model load failed: {_model_load_error}")
            return None, None

    return _tokenizer, _model


def detect_emotion(text):
    text_lower = (text or "").lower()
    if not text_lower.strip():
        return "neutral"

    anger_words = [
        "angry", "furious", "hate", "annoyed", "frustrated", "stupid",
        "ridiculous", "waste", "unfair", "blame", "can't stand"
    ]
    stress_words = [
        "urgent", "asap", "deadline", "overwhelmed", "pressure",
        "stressed", "panic", "late", "blocked", "rush"
    ]
    positive_words = [
        "thanks", "appreciate", "great", "happy", "glad",
        "excited", "support", "good", "wonderful"
    ]

    anger_score = sum(1 for word in anger_words if word in text_lower)
    stress_score = sum(1 for word in stress_words if word in text_lower)
    positive_score = sum(1 for word in positive_words if word in text_lower)

    if anger_score >= 2:
        return "angry"
    if stress_score >= 2:
        return "stressed"
    if anger_score == 1:
        return "frustrated"
    if positive_score >= 2:
        return "positive"
    return "neutral"


def run_generation(prompt, max_new_tokens=140, llm_level="advanced"):
    tokenizer, model = get_model()
    if tokenizer is None or model is None:
        return None

    generation_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": llm_level == "advanced",
        "temperature": 0.7 if llm_level == "advanced" else 0.2,
        "top_p": 0.92 if llm_level == "advanced" else 1.0,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.35
    }

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    )

    outputs = model.generate(**inputs, **generation_cfg)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def rewrite_calm_message(message, llm_level="advanced"):
    emotion = detect_emotion(message)
    prompt = f"""
You are a communication coach.
Rewrite the message to be calm, respectful, and professional.
Keep the intent, remove aggression, and keep it concise.

Return ONLY this format:
Tone: ...
Rewritten: ...

Detected emotion: {emotion}
Original message: {message}
"""

    result = run_generation(prompt, max_new_tokens=120, llm_level=llm_level)
    if result and "Rewritten:" in result:
        return result[result.find("Tone:"):] if "Tone:" in result else result

    cleaned = (message or "").replace("!", ".").replace("  ", " ").strip()
    return (
        "Tone: calm\n"
        "Rewritten: I understand the urgency and would like to collaborate on a practical next step. "
        f"{cleaned}"
    )


def mediate_conflict(side_a, side_b, shared_goal, llm_level="advanced"):
    prompt = f"""
You are a neutral AI mediator helping two people de-escalate conflict.
Provide a balanced response with practical next steps.

Respond ONLY in this format:
Summary: ...
Common Ground: ...
Mediation Plan:
1) ...
2) ...
3) ...
Suggested Message: ...

Person A position: {side_a}
Person B position: {side_b}
Shared goal: {shared_goal}
"""

    result = run_generation(prompt, max_new_tokens=180, llm_level=llm_level)
    if result and "Summary:" in result:
        return result[result.find("Summary:"):]

    return (
        "Summary: Both sides are under pressure and need clarity.\n"
        "Common Ground: They both care about successful outcomes.\n"
        "Mediation Plan:\n"
        "1) Align on immediate priorities.\n"
        "2) Separate urgent and non-urgent tasks.\n"
        "3) Agree on one clear owner and deadline.\n"
        "Suggested Message: Let us reset expectations and agree on the next concrete step today."
    )


# ----------------------------------
# AI Logic
# ----------------------------------

def solve_conflict(
    event1,
    event2,
    priority,
    email,
    llm_level="advanced"
):
    detected_emotion = detect_emotion(email)

    prompt=f"""
You are an executive assistant.
Think with high-level reasoning, risk awareness, and emotional intelligence.

Respond ONLY in this format:

Decision: ...
Reason: ...
Delegation: ...
Mediation: ...
Email: ...
Risk Level: ...
Confidence: ...

Task A: {event1}
Task B: {event2}
Priority: {priority}
Message: {email}
Detected emotion: {detected_emotion}
"""
    result = run_generation(prompt, max_new_tokens=170, llm_level=llm_level)
    if not result:
        return f"""
Decision: Prioritize {priority}

Reason: I could not load the language model, so I am applying the selected priority directly.

Delegation: Move the lower-priority event to the next available slot and notify attendees.

Mediation: Keep both sides informed with respectful communication.

Email:
Sorry, I have a scheduling conflict.
Can we move this to tomorrow?

Risk Level: medium

Confidence: medium
"""

    if "Decision:" in result:
        result=result[result.find("Decision:"):]

    if "Decision:" not in result:
        result=f"""
Decision: Prioritize {priority}

Reason: Higher priority commitment comes first.

Delegation: Reschedule lower priority task.

Mediation: Communicate trade-offs clearly and avoid blame.

Email:
Sorry, I have a scheduling conflict.
Can we move this to tomorrow?

Risk Level: low

Confidence: medium
"""

    return result


# ----------------------------------
# Entire Frontend in Python
# ----------------------------------

@app.route("/")
def home():
    frontend_path = os.path.join(PROJECT_ROOT, "frontend", "frontend.html")
    if os.path.exists(frontend_path):
        return send_file(frontend_path)

    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>LifeOS v2</title>

<style>

*{
box-sizing:border-box;
margin:0;
padding:0;
}

body{
font-family:Inter,Segoe UI,Arial,sans-serif;
background:
radial-gradient(circle at 12% 18%, rgba(34,211,238,.16), transparent 34%),
radial-gradient(circle at 85% 8%, rgba(168,85,247,.16), transparent 30%),
linear-gradient(145deg,#050816 0%,#0b1330 45%,#21114a 100%);
color:#eef2ff;
min-height:100vh;
line-height:1.5;
}

.hero{
padding:62px 26px 42px;
text-align:center;
background:
linear-gradient(
130deg,
rgba(15,23,42,.78) 0%,
rgba(30,27,75,.62) 100%
);
border-bottom:1px solid rgba(148,163,184,.2);
box-shadow:0 14px 40px rgba(10,12,30,.45);
}

.hero h1{
font-size:64px;
margin-bottom:14px;
letter-spacing:.8px;
background:linear-gradient(90deg,#e2e8f0 0%,#c4b5fd 35%,#67e8f9 100%);
-webkit-background-clip:text;
background-clip:text;
color:transparent;
}

.hero p{
font-size:20px;
opacity:.95;
max-width:980px;
margin:0 auto;
}

.badge{
display:inline-block;
padding:10px 18px;
margin:6px;
background:rgba(15,23,42,.55);
border-radius:999px;
border:1px solid rgba(125,211,252,.34);
color:#e0f2fe;
font-size:13px;
letter-spacing:.2px;
}

.container{
max-width:1500px;
margin:auto;
padding:34px;
display:grid;
grid-template-columns:1.1fr .9fr;
gap:28px;
}

.card{
background:linear-gradient(180deg,rgba(15,23,42,.84) 0%,rgba(17,24,39,.78) 100%);
backdrop-filter:blur(12px);
padding:28px;
border-radius:24px;
box-shadow:
0 22px 60px rgba(8,13,32,.42);
border:1px solid rgba(125,211,252,.16);
}

h2{
font-size:30px;
margin-bottom:20px;
color:#f8fafc;
}

label{
display:block;
margin-top:14px;
margin-bottom:8px;
font-weight:600;
opacity:.98;
color:#dbeafe;
}

input,textarea,select{
width:100%;
padding:15px 14px;
font-size:15px;
border:1px solid rgba(148,163,184,.28);
border-radius:14px;
background:rgba(2,6,23,.72);
color:#f8fafc;
outline:none;
transition:border-color .2s, box-shadow .2s, transform .2s;
}

input:focus,textarea:focus,select:focus{
border-color:#67e8f9;
box-shadow:0 0 0 3px rgba(103,232,249,.16);
transform:translateY(-1px);
}

textarea{
height:140px;
}

button{
margin-top:14px;
width:100%;
padding:14px 16px;
font-size:16px;
font-weight:700;
border:none;
border-radius:14px;
background:
linear-gradient(
96deg,
#22d3ee 0%,
#3b82f6 40%,
#8b5cf6 100%
);
color:white;
cursor:pointer;
box-shadow:0 10px 30px rgba(56,189,248,.26);
transition:transform .25s ease, box-shadow .25s ease, filter .25s ease;
}

button:hover{
transform:
translateY(-3px)
scale(1.02);
box-shadow:0 14px 34px rgba(139,92,246,.32);
filter:saturate(1.08);
}

#output{
background:linear-gradient(180deg,rgba(7,14,35,.98),rgba(14,20,40,.98));
padding:20px;
border-radius:14px;
min-height:260px;
font-size:15px;
white-space:pre-wrap;
line-height:1.65;
border:1px solid rgba(125,211,252,.2);
box-shadow:inset 0 0 0 1px rgba(255,255,255,.02);
}

.tabs{
display:grid;
grid-template-columns:repeat(4,1fr);
gap:10px;
margin-bottom:16px;
}

.tab{
padding:11px 10px;
font-size:13px;
border-radius:10px;
border:1px solid rgba(148,163,184,.26);
background:rgba(15,23,42,.8);
color:#cbd5e1;
cursor:pointer;
box-shadow:none;
}

.tab.active{
background:linear-gradient(96deg,#0ea5e9,#6366f1,#8b5cf6);
color:#f8fafc;
border-color:rgba(255,255,255,.05);
}

.panel{
display:none;
}

.panel.active{
display:block;
}

.stats{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:12px;
margin-top:14px;
}

.stat{
padding:14px 12px;
background:rgba(5,10,28,.9);
border-radius:12px;
border:1px solid rgba(125,211,252,.18);
text-align:center;
}

.muted{
opacity:.86;
font-size:13px;
color:#cbd5e1;
}

.footer{
text-align:center;
padding:20px 20px 30px;
opacity:.88;
font-size:14px;
color:#cbd5e1;
}

.app-shell{
max-width:1500px;
margin:0 auto;
padding:22px;
display:grid;
grid-template-columns:280px 1fr;
gap:16px;
height:100vh;
}

.app-sidebar{
background:rgba(15,23,42,.85);
border:1px solid rgba(148,163,184,.28);
border-radius:18px;
padding:16px;
display:flex;
flex-direction:column;
gap:12px;
}

.app-logo{
font-size:24px;
font-weight:800;
background:linear-gradient(90deg,#e2e8f0,#93c5fd,#c4b5fd);
-webkit-background-clip:text;
color:transparent;
}

.app-sub{
font-size:13px;
color:#94a3b8;
margin-top:8px;
}

.agent-menu{
display:grid;
gap:8px;
margin-top:16px;
}

.agent-item{
width:100%;
padding:11px 12px;
text-align:left;
border-radius:10px;
border:1px solid rgba(148,163,184,.28);
background:#0b1228;
color:#e2e8f0;
font-size:14px;
cursor:pointer;
}

.agent-item.active{
background:linear-gradient(95deg,rgba(59,130,246,.34),rgba(139,92,246,.34));
border-color:rgba(96,165,250,.7);
}

.mini-note{
font-size:12px;
color:#94a3b8;
line-height:1.45;
}

.new-chat-btn{
margin-top:10px;
width:100%;
padding:10px;
font-size:13px;
font-weight:700;
border-radius:10px;
border:1px solid rgba(96,165,250,.55);
background:rgba(30,64,175,.28);
color:#dbeafe;
cursor:pointer;
}

.history-title{
margin-top:12px;
font-size:12px;
color:#93c5fd;
letter-spacing:.2px;
}

.history-list{
margin-top:8px;
display:grid;
gap:7px;
max-height:220px;
overflow:auto;
padding-right:4px;
}

.history-item{
padding:8px 10px;
border-radius:9px;
border:1px solid rgba(148,163,184,.24);
background:#0b1228;
cursor:pointer;
}

.history-item.active{
border-color:rgba(96,165,250,.72);
background:rgba(59,130,246,.18);
}

.history-item-title{
font-size:12px;
color:#e2e8f0;
white-space:nowrap;
overflow:hidden;
text-overflow:ellipsis;
}

.history-item-meta{
font-size:11px;
color:#94a3b8;
margin-top:3px;
}

.chat-wrap{
background:rgba(15,23,42,.84);
border:1px solid rgba(148,163,184,.28);
border-radius:18px;
display:flex;
flex-direction:column;
overflow:hidden;
}

.chat-header{
padding:14px 16px;
display:flex;
justify-content:space-between;
align-items:center;
border-bottom:1px solid rgba(148,163,184,.25);
}

.chat-title{
font-size:18px;
font-weight:700;
}

.status-pill{
font-size:12px;
padding:4px 10px;
border-radius:999px;
background:rgba(16,185,129,.2);
color:#a7f3d0;
border:1px solid rgba(16,185,129,.36);
}

.chat-body{
padding:16px;
display:flex;
flex-direction:column;
gap:12px;
height:100%;
overflow:auto;
}

.bubble{
max-width:82%;
padding:12px 13px;
border-radius:13px;
border:1px solid rgba(148,163,184,.25);
font-size:14px;
white-space:pre-wrap;
line-height:1.5;
}

.bubble.user{
align-self:flex-end;
background:#1e293b;
}

.bubble.assistant{
align-self:flex-start;
background:#0b1228;
}

.bubble-time{
display:block;
font-size:11px;
color:#94a3b8;
margin-top:7px;
}

.chat-composer{
padding:12px;
border-top:1px solid rgba(148,163,184,.25);
display:grid;
grid-template-columns:1fr auto;
gap:8px;
}

.chat-composer textarea{
height:76px;
resize:none;
padding:11px;
background:#0b1228;
}

.send-btn{
width:108px;
margin-top:0;
}

.agent-controls{
margin-top:auto;
padding-top:10px;
border-top:1px solid rgba(148,163,184,.2);
display:grid;
gap:10px;
}

.control-label{
font-size:12px;
color:#93c5fd;
font-weight:600;
}

.mini-stats{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:6px;
}

.mini-stat{
padding:8px 6px;
background:#0b1228;
border:1px solid rgba(148,163,184,.24);
border-radius:10px;
text-align:center;
}

.mini-num{
font-size:14px;
font-weight:700;
}

.mini-key{
font-size:10px;
color:#94a3b8;
}

.chat-suggestions{
display:flex;
gap:8px;
padding:10px 12px;
border-top:1px solid rgba(148,163,184,.2);
overflow:auto;
}

.chat-suggestions button{
margin-top:0;
width:auto;
white-space:nowrap;
padding:8px 10px;
font-size:12px;
border-radius:999px;
background:#111b36;
border:1px solid rgba(148,163,184,.24);
box-shadow:none;
}

@media(max-width:900px){
.container{
grid-template-columns:1fr;
padding:16px;
}
.hero h1{
font-size:42px;
}
.tabs{
grid-template-columns:repeat(2,1fr);
}
.app-shell{
grid-template-columns:1fr;
height:auto;
}
.chat-wrap{
min-height:68vh;
}
}

</style>
</head>

<body>
<div class="app-shell">
  <aside class="app-sidebar">
    <div>
      <div class="app-logo">LifeOS v2</div>
      <div class="app-sub">Multi-agent conflict intelligence workspace</div>
    </div>

    <div class="agent-menu">
      <button class="agent-item active" onclick="selectAgent('resolve', this)">Executive Resolver</button>
      <button class="agent-item" onclick="selectAgent('emotion', this)">Emotion Analyst</button>
      <button class="agent-item" onclick="selectAgent('rewrite', this)">Calm Rewriter</button>
      <button class="agent-item" onclick="selectAgent('mediate', this)">Neutral Mediator</button>
      <button class="agent-item" onclick="window.location.href='/dashboard'" style="margin-top:12px;background:linear-gradient(90deg,#22d3ee,#8b5cf6);">🚀 Innovation Dashboard</button>
    </div>

    <button class="new-chat-btn" onclick="createNewChat()">+ New Chat</button>
    <div class="history-title">Chat History</div>
    <div class="history-list" id="historyList"></div>

    <div class="mini-note">AI Agent mode: pick a specialist and chat naturally.</div>

    <div class="agent-controls">
      <div>
        <div class="control-label">Model Level</div>
        <select id="llmLevel">
          <option value="advanced" selected>Advanced</option>
          <option value="standard">Standard</option>
        </select>
      </div>
      <div class="mini-stats">
        <div class="mini-stat"><div class="mini-num" id="totalCount">0</div><div class="mini-key">Total</div></div>
        <div class="mini-stat"><div class="mini-num" id="resolvedCount">0</div><div class="mini-key">Resolved</div></div>
        <div class="mini-stat"><div class="mini-num" id="emotionTop">-</div><div class="mini-key">Emotion</div></div>
      </div>
    </div>
  </aside>

  <main class="chat-wrap">
    <div class="chat-header">
      <div class="chat-title" id="activeTitle">Executive Resolver</div>
      <div class="status-pill">Online</div>
    </div>

    <div class="chat-body" id="chatBox">
      <div class="bubble assistant">Welcome. Pick an agent on the left and send your task below.</div>
    </div>

    <div class="chat-composer">
      <textarea id="chatInput" placeholder="Ask your selected agent..."></textarea>
      <button class="send-btn" onclick="sendMessage()">Send</button>
    </div>
    <div class="chat-suggestions">
      <button onclick="quickPrompt('Resolve: Investor meeting at 7 PM vs family dinner, priority family dinner.')">Resolve conflict</button>
      <button onclick="quickPrompt('Analyze emotion: I am very frustrated and this feels unfair.')">Analyze emotion</button>
      <button onclick="quickPrompt('Rewrite calmly: This is unacceptable, fix this now.')">Rewrite calmly</button>
      <button onclick="quickPrompt('Mediate: A wants speed, B wants quality, goal is safe delivery.')">Mediation plan</button>
    </div>
  </main>
</div>


<script>
let currentAgent = "resolve";
const AGENT_NAMES = {
resolve: "Executive Resolver",
emotion: "Emotion Analyst",
rewrite: "Calm Rewriter",
mediate: "Neutral Mediator"
};

function quickPrompt(text){
document.getElementById("chatInput").value = text;
}

const sessionsByAgent = {
resolve: [],
emotion: [],
rewrite: [],
mediate: []
};
const activeSessionByAgent = {
resolve: null,
emotion: null,
rewrite: null,
mediate: null
};

function getNow(){
return new Date();
}

function formatTime(date){
return date.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
}

function toShortTitle(text){
const clean = (text || "New chat").trim().replace(/\s+/g, " ");
return clean.length > 38 ? `${clean.slice(0, 38)}...` : clean;
}

function getActiveSession(){
const id = activeSessionByAgent[currentAgent];
return sessionsByAgent[currentAgent].find((s) => s.id === id) || null;
}

function renderHistoryList(){
const list = document.getElementById("historyList");
const sessions = [...sessionsByAgent[currentAgent]].reverse();
list.innerHTML = "";
for(const s of sessions){
const item = document.createElement("div");
item.className = `history-item${s.id === activeSessionByAgent[currentAgent] ? " active" : ""}`;
item.onclick = () => {
activeSessionByAgent[currentAgent] = s.id;
renderHistoryList();
renderChat();
};

const title = document.createElement("div");
title.className = "history-item-title";
title.innerText = s.title;
item.appendChild(title);

const meta = document.createElement("div");
meta.className = "history-item-meta";
meta.innerText = formatTime(s.createdAt);
item.appendChild(meta);

list.appendChild(item);
}
}

function renderChat(){
const chat = document.getElementById("chatBox");
chat.innerHTML = "";
const session = getActiveSession();
if(!session){
return;
}
for(const msg of session.messages){
appendMessageToDom(msg.role, msg.text, msg.timestamp);
}
chat.scrollTop = chat.scrollHeight;
}

function appendMessageToDom(role, text, timestamp){
const chat = document.getElementById("chatBox");
const node = document.createElement("div");
node.className = `bubble ${role}`;
node.innerText = text;

const time = document.createElement("span");
time.className = "bubble-time";
time.innerText = formatTime(timestamp);
node.appendChild(time);

chat.appendChild(node);
return node;
}

function appendMessage(role, text){
const session = getActiveSession();
if(!session){
return null;
}
const message = { role, text, timestamp: getNow() };
session.messages.push(message);
return appendMessageToDom(role, text, message.timestamp);
}

function createNewChat(){
const session = {
id: `${currentAgent}-${Date.now()}`,
agent: currentAgent,
title: "New chat",
createdAt: getNow(),
messages: [
{
role: "assistant",
text: `New ${AGENT_NAMES[currentAgent]} chat started. Share your request.`,
timestamp: getNow()
}
]
};
sessionsByAgent[currentAgent].push(session);
activeSessionByAgent[currentAgent] = session.id;
renderHistoryList();
renderChat();
}

function selectAgent(agent, btn){
currentAgent = agent;
document.getElementById("activeTitle").innerText = AGENT_NAMES[agent];
document.querySelectorAll(".agent-item").forEach((el) => el.classList.remove("active"));
btn.classList.add("active");
if(!activeSessionByAgent[currentAgent]){
createNewChat();
return;
}
renderHistoryList();
renderChat();
}

function buildPayloadFromMessage(message){
const llm = document.getElementById("llmLevel").value;
if(currentAgent === "resolve"){
return {
endpoint: "/resolve",
payload: {
event1: "Work Meeting",
event2: "Personal Commitment",
priority: "Personal Commitment",
email: message,
llm_level: llm
}
};
}
if(currentAgent === "emotion"){
return { endpoint: "/emotion", payload: { text: message } };
}
if(currentAgent === "rewrite"){
return { endpoint: "/rewrite", payload: { text: message, llm_level: llm } };
}
return {
endpoint: "/mediate",
payload: {
side_a: message,
side_b: "The other side needs quality and realistic timeline.",
shared_goal: "Create a balanced, respectful conflict resolution plan.",
llm_level: llm
}
};
}

function formatResponse(data){
if(data.result){ return data.result; }
if(data.emotion){
return `Emotion: ${data.emotion}\nConfidence: ${data.confidence}\nAdvice: ${data.guidance}`;
}
if(data.error){ return `Error: ${data.error}`; }
return JSON.stringify(data, null, 2);
}

async function streamAssistantResponse(text){
const node = appendMessage("assistant", "");
if(!node){
return;
}
const chunks = text.split("");
let current = "";
for(const ch of chunks){
current += ch;
node.innerHTML = "";
node.appendChild(document.createTextNode(current));
const time = document.createElement("span");
time.className = "bubble-time";
time.innerText = formatTime(getNow());
node.appendChild(time);
await new Promise((resolve) => setTimeout(resolve, 7));
}
const session = getActiveSession();
if(session){
session.messages[session.messages.length - 1].text = text;
}
}

async function sendMessage(){
const input = document.getElementById("chatInput");
const text = input.value.trim();
if(!text){ return; }

appendMessage("user", text);
input.value = "";
const session = getActiveSession();
if(session && session.title === "New chat"){
session.title = toShortTitle(text);
}
renderHistoryList();
const loadingNode = appendMessage("assistant", "Thinking...");

const req = buildPayloadFromMessage(text);
try{
const res = await fetch(req.endpoint, {
method: "POST",
headers: {"Content-Type":"application/json"},
body: JSON.stringify(req.payload)
});
const data = await res.json();
if(loadingNode){ loadingNode.remove(); }
const formatted = formatResponse(data);
await streamAssistantResponse(formatted);
loadHistory();
}catch(err){
if(loadingNode){ loadingNode.remove(); }
appendMessage("assistant", `Request failed: ${err}`);
}
}

async function loadHistory(){
const res = await fetch("/history");
const data = await res.json();
document.getElementById("totalCount").innerText = data.total;
document.getElementById("resolvedCount").innerText = data.resolved;
document.getElementById("emotionTop").innerText = data.top_emotion || "-";
}

document.getElementById("chatInput").addEventListener("keydown", (e) => {
if(e.key === "Enter" && !e.shiftKey){
e.preventDefault();
sendMessage();
}
});

createNewChat();
loadHistory();
</script>

</body>
</html>
"""


@app.route(
"/resolve",
methods=["POST"]
)
def resolve():
    data = request.get_json(silent=True) or {}

    required_fields = ("event1", "event2", "priority", "email")
    missing_fields = [field for field in required_fields if not str(data.get(field, "")).strip()]
    if missing_fields:
        return jsonify(
            {
                "error": "Missing required fields",
                "missing": missing_fields
            }
        ), 400

    llm_level = data.get("llm_level", "advanced")

    try:
        result=solve_conflict(
            data["event1"],
            data["event2"],
            data["priority"],
            data["email"],
            llm_level
        )
    except Exception as exc:
        return jsonify(
            {
                "error": "Failed to generate decision",
                "details": str(exc)
            }
        ), 500

    _history.append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event1": data["event1"],
            "event2": data["event2"],
            "priority": data["priority"],
            "emotion": detect_emotion(data["email"]),
            "result": result
        }
    )
    if len(_history) > 200:
        _history.pop(0)

    return jsonify({"result": result})


@app.route("/emotion", methods=["POST"])
def emotion():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    emotion_label = detect_emotion(text)
    confidence = "medium"
    if emotion_label in ("angry", "stressed"):
        confidence = "high"
    elif emotion_label == "neutral":
        confidence = "low"

    guidance = (
        "Use short, respectful sentences and focus on one next action."
        if emotion_label in ("angry", "frustrated", "stressed")
        else "Maintain clarity and collaborative tone."
    )
    return jsonify({
        "emotion": emotion_label,
        "confidence": confidence,
        "guidance": guidance
    })


@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "text or message is required"}), 400

    llm_level = data.get("llm_level", "advanced")
    result = rewrite_calm_message(text, llm_level=llm_level)
    return jsonify({"result": result})


@app.route("/mediate", methods=["POST"])
def mediate():
    data = request.get_json(silent=True) or {}
    side_a = str(data.get("side_a", "")).strip()
    side_b = str(data.get("side_b", "")).strip()
    shared_goal = str(data.get("shared_goal", "")).strip()

    if not side_a or not side_b or not shared_goal:
        return jsonify(
            {"error": "side_a, side_b, and shared_goal are required"}
        ), 400

    llm_level = data.get("llm_level", "advanced")
    result = mediate_conflict(side_a, side_b, shared_goal, llm_level=llm_level)
    return jsonify({"result": result})


@app.route("/history")
def history():
    emotion_counts = {}
    for item in _history:
        emotion = item.get("emotion", "unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    top_emotion = None
    if emotion_counts:
        top_emotion = sorted(
            emotion_counts.items(),
            key=lambda pair: pair[1],
            reverse=True
        )[0][0]

    return jsonify(
        {
            "total": len(_history),
            "resolved": len(_history),
            "top_emotion": top_emotion,
            "recent": _history[-10:]
        }
    )


@app.route("/health")
def health():
    status = "ok" if _model_load_error is None else "degraded"
    return jsonify(
        {
            "status": status,
            "model_name": MODEL_NAME,
            "model_ready": _tokenizer is not None and _model is not None,
            "model_error": _model_load_error
        }
    )


@app.route("/favicon.ico")
def favicon():
    # Avoid noisy 404s in browser console when no icon is provided.
    return ("", 204)


@app.route("/api/rewards/summary", methods=["GET"])
def api_rewards_summary():
    rows = _load_reward_artifacts()
    summary = _reward_summary(rows)
    return jsonify(summary)


@app.route("/api/rewards/timeline", methods=["GET"])
def api_rewards_timeline():
    rows = _load_reward_artifacts()
    if not rows:
        return jsonify({"available": False, "timeline": []})

    timeline = []
    for idx, row in enumerate(rows):
        try:
            reward = float(row.get("reward", 0))
        except Exception:
            reward = 0.0
        epoch = row.get("epoch", idx)
        timeline.append({
            "epoch": int(epoch) if str(epoch).isdigit() else idx,
            "reward": reward,
            "persona": row.get("persona", "unknown"),
            "policy": row.get("policy", "unknown")
        })

    return jsonify({
        "available": True,
        "points": len(timeline),
        "timeline": timeline
    })


@app.route("/api/judging/readiness", methods=["GET"])
def api_judging_readiness():
    rows = _load_reward_artifacts()
    reward_data = _reward_summary(rows)
    readiness = {
        "openenv_usage": True,
        "trl_colab_script": True,
        "reward_evidence_available": bool(reward_data.get("available")),
        "mini_blog_or_video_published": False,
        "hf_space_hosted": False
    }
    return jsonify({
        "readiness": readiness,
        "reward_summary": reward_data,
        "notes": [
            "Publish mini-blog or <2 min video link to mark ready.",
            "Host app on Hugging Face Spaces to complete minimum requirements."
        ]
    })


@app.route("/api/agent/capabilities", methods=["GET"])
def api_agent_capabilities():
    return jsonify({
        "core_modules": [
            "Conflict Resolution Agent",
            "Calendar Agent",
            "Negotiation Agent",
            "Email Reply Agent",
            "Delegation Agent",
            "Memory Agent",
            "Emotion Detection Agent"
        ],
        "advanced_features": [
            "Emotion detection module",
            "Response rewriter",
            "Mediation mode",
            "Conflict progress dashboard",
            "Conflict history memory",
            "Predictive conflict handling",
            "Schema drift adaptation",
            "Consumer workflow automation"
        ]
    })


# ----------------------------------
# OpenEnv Innovation Dashboard
# ----------------------------------

def record_episode(episode_data):
    """Record an episode for dashboard visualization."""
    episode_data["timestamp"] = datetime.now().isoformat()
    _episode_history.append(episode_data)
    # Keep last 100 episodes
    if len(_episode_history) > 100:
        _episode_history.pop(0)


@app.route("/dashboard")
def dashboard():
    """Innovation Dashboard - Visualizes per-episode data for judges."""
    
    # Aggregate statistics
    total_episodes = len(_episode_history)
    
    # Persona distribution
    persona_counts = defaultdict(int)
    for ep in _episode_history:
        persona = ep.get("persona", {}).get("name", "Unknown")
        persona_counts[persona] += 1
    
    # Drift events
    drift_counts = defaultdict(int)
    for ep in _episode_history:
        policy = ep.get("policy", {})
        drift = policy.get("version", "unknown")
        drift_counts[drift] += 1
    
    # Tool usage stats
    tool_stats = defaultdict(int)
    for ep in _episode_history:
        tools = ep.get("tool_results", {})
        for tool_name in tools:
            tool_stats[tool_name] += 1
    
    # Reward stats
    rewards = [ep.get("total_reward", 0) for ep in _episode_history]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    max_reward = max(rewards) if rewards else 0
    min_reward = min(rewards) if rewards else 0
    
    # Recent episodes (last 10)
    recent = _episode_history[-10:] if _episode_history else []
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>OpenEnv Innovation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{
    font-family:'Inter',system-ui,sans-serif;
    background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);
    color:#f1f5f9; min-height:100vh; padding:24px;
}}
.dashboard-header {{
    text-align:center; padding:32px 0;
    background:linear-gradient(90deg,#22d3ee,#8b5cf6,#ec4899);
    -webkit-background-clip:text; background-clip:text;
    color:transparent;
}}
.dashboard-header h1 {{ font-size:48px; margin-bottom:8px; }}
.dashboard-header p {{ opacity:0.8; font-size:18px; }}

.stats-grid {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:20px;
    margin:32px 0;
}}
.stat-card {{
    background:rgba(15,23,42,0.9); border:1px solid rgba(103,232,249,0.2);
    border-radius:16px; padding:24px; text-align:center;
}}
.stat-card .value {{ font-size:42px; font-weight:700; color:#22d3ee; }}
.stat-card .label {{ font-size:14px; opacity:0.7; margin-top:8px; }}

.test-btn {{
    display:inline-block; margin:20px auto; text-align:center;
}}
.test-btn button {{
    background:linear-gradient(90deg,#22d3ee,#8b5cf6);
    padding:14px 32px; font-size:16px; border-radius:12px;
    cursor:pointer; border:none; color:white; font-weight:600;
    box-shadow:0 4px 20px rgba(139,92,246,0.4);
}}
.test-btn button:hover {{ transform:scale(1.05); }}

.charts-row {{
    display:grid; grid-template-columns:1fr 1fr; gap:24px; margin:24px 0;
}}
.chart-card {{
    background:rgba(15,23,42,0.9); border:1px solid rgba(103,232,249,0.15);
    border-radius:16px; padding:20px;
}}
.chart-card h3 {{ font-size:18px; margin-bottom:16px; color:#94a3b8; }}

.episode-table {{
    width:100%; border-collapse:collapse; margin-top:24px;
    background:rgba(15,23,42,0.9); border-radius:16px; overflow:hidden;
}}
.episode-table th {{
    background:rgba(30,41,59,0.9); padding:16px; text-align:left;
    font-size:13px; text-transform:uppercase; letter-spacing:0.5px;
}}
.episode-table td {{
    padding:14px 16px; border-top:1px solid rgba(51,65,85,0.5);
}}
.episode-table tr:hover {{ background:rgba(30,41,59,0.5); }}

.badge {{
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:12px; font-weight:600;
}}
.badge-persona {{ background:rgba(139,92,246,0.2); color:#c4b5fd; }}
.badge-drift {{ background:rgba(34,211,238,0.2); color:#67e8f9; }}
.badge-reward {{ background:rgba(34,197,94,0.2); color:#4ade80; }}
.badge-negative {{ background:rgba(239,68,68,0.2); color:#f87171; }}

.delegation-graph {{
    display:flex; justify-content:center; gap:16px; flex-wrap:wrap;
    margin:20px 0;
}}
.delegation-node {{
    background:linear-gradient(135deg,#1e293b,#334155);
    border:1px solid rgba(103,232,249,0.3); border-radius:12px;
    padding:16px 24px; text-align:center;
}}
.delegation-node .name {{ font-weight:600; color:#22d3ee; }}
.delegation-node .action {{ font-size:13px; opacity:0.8; margin-top:4px; }}

.empty-state {{
    text-align:center; padding:60px; opacity:0.6;
}}
.empty-state h2 {{ font-size:24px; margin-bottom:12px; }}
</style>
</head>
<body>

<div class="dashboard-header">
    <h1>🚀 OpenEnv Innovation Dashboard</h1>
    <p>Real-time episode visualization for judges • Hackathon Demo</p>
</div>

<div class="stats-grid">
    <div class="stat-card">
        <div class="value">{total_episodes}</div>
        <div class="label">Total Episodes</div>
    </div>
    <div class="stat-card">
        <div class="value">{avg_reward:.1f}</div>
        <div class="label">Avg Reward</div>
    </div>
    <div class="stat-card">
        <div class="value">{max_reward}</div>
        <div class="label">Max Reward</div>
    </div>
    <div class="stat-card">
        <div class="value">{len(tool_stats)}</div>
        <div class="label">Tools Used</div>
    </div>
</div>

<div class="test-btn">
    <button onclick="runTestEpisodes()">🚀 Run 5 Test Episodes</button>
    <div id="testStatus" style="margin-top:12px; font-size:14px; opacity:0.8;"></div>
</div>

<script>
async function runTestEpisodes() {{
    const statusDiv = document.getElementById('testStatus');
    statusDiv.innerHTML = 'Running episodes...';
    for (let i = 0; i < 5; i++) {{
        await fetch('/api/run-episode', {{method: 'POST'}});
        statusDiv.innerHTML = `Episode ${{i+1}}/5 completed`;
    }}
    statusDiv.innerHTML = '✅ All episodes generated! Reloading...';
    setTimeout(() => window.location.reload(), 800);
}}
</script>
"""
    
    # Add charts if we have data
    if total_episodes > 0:
        # Persona chart data
        persona_labels = list(persona_counts.keys())
        persona_data = list(persona_counts.values())
        
        # Drift chart data
        drift_labels = list(drift_counts.keys())
        drift_data = list(drift_counts.values())
        
        html += f"""
<div class="charts-row">
    <div class="chart-card">
        <h3>👤 Persona Distribution</h3>
        <canvas id="personaChart"></canvas>
    </div>
    <div class="chart-card">
        <h3>📜 Policy Drift Events</h3>
        <canvas id="driftChart"></canvas>
    </div>
</div>

<div class="chart-card" style="margin:24px 0;">
    <h3>📈 Reward Over Time</h3>
    <canvas id="rewardChart" height="80"></canvas>
</div>

<h2 style="margin:32px 0 16px;">Recent Episodes</h2>
<table class="episode-table">
<thead>
<tr>
    <th>Episode</th>
    <th>Persona</th>
    <th>Drift Event</th>
    <th>Delegation</th>
    <th>Tools</th>
    <th>Reward</th>
</tr>
</thead>
<tbody>
"""
        
        for i, ep in enumerate(recent):
            persona = ep.get("persona", {})
            policy = ep.get("policy", {})
            delegation = ep.get("delegation", {})
            tools = ep.get("tool_results", {})
            reward = ep.get("total_reward", 0)
            
            persona_name = persona.get("name", "N/A")
            drift = policy.get("version", "N/A")
            
            # Delegation nodes
            del_nodes = ""
            for agent, action in list(delegation.items())[:3]:
                short_action = action[:40] + "..." if len(action) > 40 else action
                del_nodes += f'<div class="delegation-node"><div class="name">{agent}</div><div class="action">{short_action}</div></div>'
            
            # Tool badges
            tool_badges = ", ".join(tools.keys()) if tools else "none"
            
            reward_badge = "positive" if reward > 0 else "negative"
            
            html += f"""
<tr>
    <td>#{i+1}</td>
    <td><span class="badge badge-persona">{persona_name}</span></td>
    <td><span class="badge badge-drift">{drift}</span></td>
    <td>{len(delegation)} agents</td>
    <td>{tool_badges}</td>
    <td><span class="badge badge-{reward_badge}">{reward}</span></td>
</tr>
"""
        
        html += """
</tbody>
</table>

<script>
// Persona Distribution Chart
new Chart(document.getElementById('personaChart'), {
    type: 'doughnut',
    data: {
        labels: """ + json.dumps(persona_labels) + """,
        datasets: [{
            data: """ + json.dumps(persona_data) + """,
            backgroundColor: ['#8b5cf6','#22d3ee','#ec4899','#f59e0b','#10b981'],
            borderWidth: 0
        }]
    },
    options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } } }
});

// Drift Events Chart
new Chart(document.getElementById('driftChart'), {
    type: 'bar',
    data: {
        labels: """ + json.dumps(drift_labels) + """,
        datasets: [{
            label: 'Episodes',
            data: """ + json.dumps(drift_data) + """,
            backgroundColor: '#22d3ee',
            borderRadius: 8
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } }
    }
});

// Reward Over Time Chart
new Chart(document.getElementById('rewardChart'), {
    type: 'line',
    data: {
        labels: """ + json.dumps([f"Ep {i+1}" for i in range(len(_episode_history))]) + """,
        datasets: [{
            label: 'Total Reward',
            data: """ + json.dumps(rewards) + """,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139,92,246,0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#8b5cf6'
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
            y: { beginAtZero: true, ticks: { color: '#94a3b8' } },
            x: { ticks: { color: '#94a3b8' } }
        }
    }
});
</script>
"""
    else:
        html += """
<div class="empty-state">
    <h2>🎯 No Episodes Yet</h2>
    <p>Run some episodes to see the dashboard populate with data.</p>
    <p>Use the /api/run-episode endpoint to generate episodes.</p>
</div>
"""
    
    html += """
<div style="text-align:center; padding:40px; opacity:0.6; font-size:14px;">
    <p>OpenEnv Innovation Dashboard • Powered by LifeOS v2</p>
</div>
</body>
</html>
"""
    return html


@app.route("/api/run-episode", methods=["POST"])
def api_run_episode():
    """Run a single episode and record it for the dashboard."""
    from openenv_env.env import LifeOSEnv
    from openenv_env.personas import PERSONAS
    
    # Load scenarios
    scenarios_path = os.path.join(PROJECT_ROOT, "demo_scenarios.json")
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    
    env = LifeOSEnv(scenarios)
    state = env.reset()
    
    # Simulate an action
    action = "reschedule meeting to tomorrow"
    state, reward, done, info = env.step(action)
    
    # Record episode for dashboard
    episode_data = {
        "scenario": info.get("scenario", {}),
        "persona": info.get("persona", {}),
        "policy": info.get("policy", {}),
        "prediction": info.get("prediction", {}),
        "workflow": info.get("workflow", []),
        "plan": info.get("plan", []),
        "delegation": info.get("delegation", {}),
        "tool_results": info.get("tool_results", {}),
        "reward_breakdown": info.get("reward_breakdown", {}),
        "total_reward": reward,
        "action": action
    }
    record_episode(episode_data)
    
    return jsonify({
        "status": "success",
        "episode_recorded": True,
        "reward": reward,
        "breakdown": info.get("reward_breakdown", {}),
        "dashboard_url": "/dashboard"
    })


@app.route("/api/episodes", methods=["GET"])
def api_episodes():
    """Get all recorded episodes."""
    return jsonify({
        "total": len(_episode_history),
        "episodes": _episode_history
    })


@app.route("/api/episode/latest", methods=["GET"])
def api_episode_latest():
    """Get the most recent episode."""
    if not _episode_history:
        return jsonify({"error": "No episodes recorded"})
    return jsonify(_episode_history[-1])


if __name__=="__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(
        debug=debug_mode,
        use_reloader=False,
        host="0.0.0.0",
        port=5000
    )