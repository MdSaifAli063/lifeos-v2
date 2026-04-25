import os
from datetime import datetime

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# ----------------------------------
# Model
# ----------------------------------

MODEL_NAME = os.getenv("LIFEOS_MODEL_NAME", "google/flan-t5-base")
_tokenizer = None
_model = None
_model_load_error = None
_history = []


def get_model():
    global _tokenizer, _model, _model_load_error

    if _tokenizer is None or _model is None:
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
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

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


if __name__=="__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(
        debug=debug_mode,
        use_reloader=False,
        host="0.0.0.0",
        port=5000
    )