# AetherMind: Building an AI Agent for Real-Life Conflict Resolution

AetherMind is an OpenEnv-based AI agent designed for realistic personal workflow conflicts: overlapping meetings, family-vs-work decisions, difficult email replies, and negotiation trade-offs.

## The Core Problem

Most assistants work well only in clean conditions. Real users face:
- competing priorities,
- emotional communication,
- changing policies/API contracts (schema drift),
- and repeated decisions that require memory of past preferences.

AetherMind is built specifically for this messy, real-world setting.

## What We Built

We created an OpenEnv-compliant environment and agent stack that includes:
- memory-aware context and personalization,
- schema/policy drift simulation,
- multi-step planning and delegation,
- reward-driven behavior optimization.

Main modules:
- `openenv_env/env.py`
- `openenv_env/memory.py`
- `openenv_env/schema_manager.py`
- `openenv_env/workflow_engine.py`
- `openenv_env/reward.py`

## Agent Features in the Demo

AetherMind command center supports:
- Conflict Resolver
- Emotion Detection
- Response Rewriter (angry -> calm)
- Mediation Mode
- Conflict History
- What-If Simulation (risk/benefit outcomes)
- Ethical Decision Filter
- Feedback Reinforcement Loop

Frontend and APIs are fully connected for live interactions.

## Training Pipeline (HF TRL PPO)

We train with PPO using Hugging Face TRL:
- full training script: `training/train_ppo.py`
- Colab-friendly script: `training/train_ppo_colab.py`

Example run:

```bash
python training/train_ppo.py --mode fast --output_dir training_outputs
```

Colab run also produces:
- `training_outputs/reward_log_colab.json`
- `training_outputs/reward_log_colab.csv`

## Evidence of Improvement

Training outputs track epoch-level rewards and behavior signals, showing progress in:
- priority alignment,
- policy compliance,
- delegation/tool execution quality.

These artifacts are used directly in pitch/demo to show measurable learning progress.

## Why This Matters

AetherMind is a practical pattern for personal AI agents:
- grounded in realistic environment simulation,
- adaptive under schema drift,
- emotionally aware in communication,
- and accountable through reward-based evaluation.

It moves beyond static assistants toward robust, decision-capable AI agents for daily life.

## Latest Production Updates

For final demo/pitch readiness, we updated deployment and runtime behavior:
- default inference model is `google/flan-t5-large` (`LIFEOS_MODEL_NAME`),
- fast UX mode is now the default in the app (`standard` tier) for low-latency live demos,
- heavy routes still support `advanced` / `high` when deeper generation is needed,
- backend model loading is thread-safe to avoid duplicate loads under concurrent requests,
- frontend now has stronger network error guidance for all feature panels.

## Hugging Face Spaces Deployment Notes

- Space SDK: **Docker**
- Required README metadata block is included in `README.md`
- App serves correctly on Space port via `PORT` env handling
- Key health endpoint: `/health`

