# 🧠 AetherMind — AI Agent for Conflict Resolution

## 🚀 Overview
AetherMind is an AI-powered autonomous agent trained using reinforcement learning to handle real-life conflicts such as meeting clashes, personal commitments, and difficult email replies.

## 🎤 One-Line Strong Pitch
AetherMind is an emotionally intelligent conflict agent that detects tone, rewrites heated messages calmly, tracks conflict history, and guides both sides toward resolution.

---

## ❗ Problem
Existing productivity assistants fail to:
- Handle conflicting priorities
- Adapt to changing data formats (schema drift)
- Learn from past decisions

---

## 💡 Solution
We built a custom OpenEnv environment where an AI agent:
- Resolves task conflicts
- Manages workflows (multi-step reasoning)
- Learns from memory
- Adapts to schema drift

---

## 🔥 Key Features
- 🧠 Memory-based decision making  
- 🔁 Multi-step workflow engine  
- 🔄 Schema drift simulation (API/data changes)  
- 🤖 Reinforcement Learning (TRL PPO)  
- 🧩 Emotion detection module  
- 🗣️ Response rewriter (angry message -> calm version)  
- 📊 Conflict progress dashboard  
- 🤝 Mediation mode (neutral AI facilitator)  
- 📁 Conflict history tracking  
- 🖥️ Live demo interface  

---

## ⚙️ Tech Stack
- Python  
- Transformers  
- TRL (Reinforcement Learning)  
- OpenEnv  
- Gradio  
- HuggingFace Spaces  

---

## 🧪 Training
We trained the model using PPO to maximize:
- Correct priority decisions  
- Smart rescheduling  
- Polite communication  

---

## 📊 Results
| Stage | Behavior |
|------|--------|
| Before Training | Random ❌ |
| After Training | Structured & intelligent ✅ |

---

## 🌍 Demo
👉 Live HuggingFace Space: [Add your link]

---

## 🎥 Demo Video
👉 [Add YouTube link]

---

## ✅ Submission Readiness

### Minimum Requirements
- OpenEnv usage: implemented in `openenv_env/env.py` and listed in `requirements.txt`.
- Training with HF TRL:
  - Full pipeline: `training/train_ppo.py`
  - Minimal Colab-friendly script: `training/train_ppo_colab.py`
- Mini-blog / mini-video (<2 min): add your final published link.
- Hosted OpenEnv app on Hugging Face Spaces: add your Space URL.

Use `SUBMISSION_CHECKLIST.md` for final pre-submission verification.

---

## 📈 Show Reward Improvement (for judging)

Run fast training and generate reward logs:

```bash
python training/train_ppo.py --mode fast --output_dir training_outputs
```

This creates:
- `training_outputs/reward_log.csv`
- `training_outputs/reward_log.json`

You can use these artifacts during your pitch to show training progress.

---

## 🧪 Colab Minimal Training (HF TRL)

```bash
python training/train_ppo_colab.py
```

This script is intentionally minimal for quick demo/training walkthroughs.

---

## 📦 Installation

```bash
pip install -r requirements.txt