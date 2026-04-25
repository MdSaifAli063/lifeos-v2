# AetherMind: Autonomous Conflict-Resolution Agent in OpenEnv

AetherMind is an autonomous AI agent that resolves real-world conflicts such as overlapping commitments, difficult email responses, and negotiation trade-offs.

## The Problem

Most assistants fail when:
- priorities collide (work vs personal commitments),
- rules change unexpectedly (schema/policy drift),
- memory of prior preferences is required.

## Our Environment Innovation

We built an OpenEnv-compliant environment with:
- memory-aware decision context,
- schema drift simulation,
- multi-step workflow planning,
- delegation/tool execution,
- coherent reward shaping for better behavior.

Core implementation:
- `openenv_env/env.py`
- `openenv_env/memory.py`
- `openenv_env/schema_manager.py`
- `openenv_env/workflow_engine.py`
- `openenv_env/reward.py`

## Agent Capabilities

- Emotion detection module
- Response rewriter (angry -> calm professional)
- Mediation mode (neutral facilitation)
- Conflict history tracking
- Predictive and policy-aware conflict handling

## Training Setup (HF TRL PPO)

We train using HF TRL PPO:
- Full pipeline: `training/train_ppo.py`
- Minimal Colab script: `training/train_ppo_colab.py`

Run:

```bash
python training/train_ppo.py --mode fast --output_dir training_outputs
```

This generates reward logs:
- `training_outputs/reward_log.csv`
- `training_outputs/reward_log.json`

## Observable Improvement

During training, reward metrics show improved alignment in:
- priority handling,
- policy compliance,
- tool/delegation behavior.

## Demo Experience

The web command center (`frontend/frontend.html`) visualizes:
- central agent-brain reasoning,
- collaborating agent nodes,
- progress and metrics,
- conflict resolution outputs in real time.

## Why It Matters

AetherMind demonstrates a practical “Personal AI Agent Operating System” pattern:
- environment-grounded reasoning,
- memory and adaptation under drift,
- measurable RL improvement,
- human-friendly conflict outcomes.

