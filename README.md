---
title: AetherMind
emoji: "🧠"
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app/app.py
pinned: false
---

# 🧠 AetherMind — AI Agent for Conflict Resolution

## 🚀 Overview
AetherMind is an AI-powered agent built on OpenEnv to resolve realistic personal workflow conflicts such as overlapping commitments, emotional messages, and scheduling trade-offs.

## 🎤 One-Line Pitch
AetherMind detects tone, rewrites high-friction communication, mediates decisions, and learns from outcomes with reinforcement learning.

---

## ❗ Problem
Most assistants fail when:
- priorities collide (work vs personal life),
- schema/policy rules change,
- memory of user preferences is required,
- communication becomes emotionally charged.

---

## 💡 Solution
We built a custom OpenEnv environment where the agent:
- resolves conflicts with priority-aware reasoning,
- adapts to schema drift and policy shifts,
- uses memory context and workflow planning,
- improves behavior through PPO reward optimization.

---

## 🔥 Key Features
- Conflict Resolver
- Emotion Detection
- Response Rewriter (angry -> calm)
- Mediation Mode
- Conflict History
- What-If Simulation (risk/benefit outcomes)
- Ethical Decision Filter
- Feedback Reinforcement Loop
- Live dashboard with agent reasoning visualization

---

## ⚙️ Tech Stack
- Python
- Flask
- OpenEnv
- Transformers
- TRL (PPO)
- Hugging Face Spaces (Docker)

---

## 🧪 Training (HF TRL PPO)

### Full training
```bash
python training/train_ppo.py --mode fast --output_dir training_outputs
```

Generates:
- `training_outputs/reward_log.csv`
- `training_outputs/reward_log.json`

### Colab-friendly training
```bash
python training/train_ppo_colab.py --mode fast --epochs 8
```

Generates:
- `training_outputs/reward_log_colab.json`
- `training_outputs/reward_log_colab.csv`

Colab step-by-step guide:
- `COLAB_NOTEBOOK_FLOW.md`

---

## 📊 Reward Improvement Evidence
We track reward progression and behavioral quality per epoch (priority alignment, policy compliance, and tool execution) and use these artifacts directly in judging demo/pitch.

---

## 🌍 Live Demo Links
- Hugging Face Space: **[Add Space URL]**
- Colab Notebook: **[Add Colab URL]**
- Mini-blog or <2 min video: **[Add link]**

---

## ✅ Submission Readiness

Minimum requirements mapping:
- OpenEnv usage: `openenv_env/env.py`
- HF TRL training scripts:
  - `training/train_ppo.py`
  - `training/train_ppo_colab.py`
- Reward artifacts in `training_outputs/`
- Deployment guide: `HF_SPACES_DEPLOY.md`
- Checklist: `SUBMISSION_CHECKLIST.md`
- Detailed readiness: `JUDGING_READINESS.md`

---

## 📦 Installation
```bash
pip install -r requirements.txt
python app/app.py
```

Open:
- `http://localhost:5000`

---

## 📁 Repository Structure
- `app/` - Flask backend and APIs
- `frontend/` - Dashboard UI
- `openenv_env/` - OpenEnv environment, reward logic, memory, schema drift
- `training/` - PPO training scripts
- `training_outputs/` - Reward logs and artifacts
- `utils/` - Prompt utilities