# LifeOS v2 Submission Checklist

Use this as your final pre-submission checklist.

## Minimum Requirements

- [x] **OpenEnv usage in project**
  - Environment: `openenv_env/env.py`
  - Dependency listed in `requirements.txt` (`openenv`)

- [x] **Minimal training script using HF TRL**
  - Main script: `training/train_ppo.py`
  - Colab-friendly minimal script: `training/train_ppo_colab.py`

- [ ] **Mini-blog on Hugging Face OR <2 min YouTube mini-video**
  - Draft prepared: `MINI_BLOG_DRAFT.md`
  - Add published link here: `<PASTE_LINK>`

- [ ] **OpenEnv environment hosted on Hugging Face Spaces**
  - Deployment guide: `HF_SPACES_DEPLOY.md`
  - Add Space URL here: `<PASTE_SPACE_URL>`

## First-Round Judging Alignment

### 1) Environment Innovation (40%)
- OpenEnv environment includes conflict resolution with:
  - Memory (`openenv_env/memory.py`)
  - Schema drift (`openenv_env/schema_manager.py`)
  - Workflow logic (`openenv_env/workflow_engine.py`)

### 2) Storytelling (30%)
- Use a 3-minute flow:
  1. Problem (conflicting priorities in real life)
  2. Environment design (memory + schema drift + reward)
  3. Agent behavior before/after PPO
  4. Live UI demo (`app/app.py`)

### 3) Showing Improvement in Rewards (20%)
- Run training and collect reward logs:
  - `python training/train_ppo.py --mode fast --output_dir training_outputs`
- Artifacts generated:
  - `training_outputs/reward_log.csv`
  - `training_outputs/reward_log.json`

### 4) Reward + Pipeline Setup (10%)
- Reward function defined in `openenv_env/reward.py`
- PPO training pipeline in `training/train_ppo.py`
- Fast/default/high profiles included for reproducible demos

## Quick Pitch Materials (ready to present)

- One-line pitch:
  - *LifeOS v2 is an emotionally intelligent AI agent that resolves scheduling conflicts, rewrites tense communication, and mediates outcomes using an OpenEnv-based RL environment.*

- Demo script:
  1. Start app: `python app/app.py`
  2. Show `Executive Resolver`
  3. Show `Emotion Analyst`
  4. Show `Calm Rewriter`
  5. Show `Neutral Mediator`
  6. Mention reward improvement from `training_outputs/reward_log.csv`
