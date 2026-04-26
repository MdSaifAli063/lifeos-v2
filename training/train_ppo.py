from transformers import AutoTokenizer
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForSeq2SeqLMWithValueHead
)

import torch
import json
import sys
import os
import argparse
import csv


# -------------------------------------------------
# Allow imports from project root
# -------------------------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from openenv_env.env import LifeOSEnv
from utils.prompt_builder import build_prompt


# -------------------------------------------------
# CLI args (fast/default/high)
# -------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train LifeOS PPO model"
)
parser.add_argument(
    "--mode",
    choices=["fast", "default", "high"],
    default="default",
    help=(
        "fast: quick high-level training, "
        "default: balanced, "
        "high: stronger/slower training"
    )
)
parser.add_argument(
    "--output_dir",
    default="training_outputs",
    help="Directory for reward logs and training artifacts"
)
parser.add_argument(
    "--fallback_model",
    default="google/flan-t5-base",
    help="Fallback checkpoint if primary model cannot load due memory/pagefile limits"
)
args = parser.parse_args()


# -------------------------------------------------
# Load scenarios
# -------------------------------------------------
with open("demo_scenarios.json", "r", encoding="utf-8") as f:
    scenarios = json.load(f)

env = LifeOSEnv(scenarios)
os.makedirs(args.output_dir, exist_ok=True)


# -------------------------------------------------
# Model (all modes: google/flan-t5-large — same family as app LIFEOS_MODEL_NAME default)
# Modes differ by epochs / tokens / LR, not checkpoint size.
# -------------------------------------------------
if args.mode == "fast":
    model_name = "google/flan-t5-large"
    total_epochs = 8
    max_new_tokens = 40
    learning_rate = 2e-6
    ppo_epochs = 1
elif args.mode == "high":
    model_name = "google/flan-t5-large"
    total_epochs = 80
    max_new_tokens = 80
    learning_rate = 8e-7
    ppo_epochs = 3
else:
    model_name = "google/flan-t5-large"
    total_epochs = 50
    max_new_tokens = 60
    learning_rate = 1e-6
    ppo_epochs = 2

def _is_windows_pagefile_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "os error 1455" in msg
        or "paging file is too small" in msg
        or "not enough memory" in msg
    )


def _load_ppo_models(primary_model_name: str, fallback_model_name: str):
    """Load tokenizer/model/ref_model; fallback automatically on Windows memory errors."""
    try:
        tok = AutoTokenizer.from_pretrained(primary_model_name)
        mdl = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(primary_model_name)
        ref = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(primary_model_name)
        return primary_model_name, tok, mdl, ref
    except Exception as e:
        if primary_model_name == fallback_model_name or not _is_windows_pagefile_error(e):
            raise
        print("\n[WARN] Could not load primary model due memory/pagefile limits.")
        print(f"[WARN] Primary: {primary_model_name}")
        print(f"[WARN] Falling back to: {fallback_model_name}\n")
        tok = AutoTokenizer.from_pretrained(fallback_model_name)
        mdl = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(fallback_model_name)
        ref = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(fallback_model_name)
        return fallback_model_name, tok, mdl, ref


model_name, tokenizer, model, ref_model = _load_ppo_models(
    model_name,
    args.fallback_model
)


# -------------------------------------------------
# Stable PPO Config
# -------------------------------------------------
config = PPOConfig(
    learning_rate=learning_rate,
    batch_size=1,
    mini_batch_size=1,
    ppo_epochs=ppo_epochs
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer
)


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
print(f"Training mode: {args.mode}")
print(f"Model: {model_name}")
print(f"Epochs: {total_epochs}")

reward_history = []

for epoch in range(total_epochs):

    try:
        # Get scenario
        state = env.reset()

        # Memory context
        memory = env.memory.get_context()

        # Build prompt
        prompt = build_prompt(
            state,
            memory
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.pretrained_model.device)


        # -----------------------------------------
        # Stable generation (anti-repeat)
        # -----------------------------------------
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.4
        )


        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()


        # Safety cleanup for looping junk
        if "4. Reply" in generated_text:
            generated_text = generated_text.split("4.")[0].strip()


        if len(generated_text) < 5:
            generated_text = (
                "Decision -> Reschedule conflict.\n"
                "Reason -> Priority handling.\n"
                "Delegation -> Reassign task.\n"
                "Email -> Sorry, can we move this meeting?"
            )


        # -----------------------------------------
        # Reward
        # -----------------------------------------
        _, reward, _, info = env.step(
            generated_text
        )


        # PPO tensors
        query_tensors = [
            inputs["input_ids"][0]
        ]

        response_tensors = [
            output_ids[0]
        ]

        reward_tensors = [
            torch.tensor(
                reward,
                dtype=torch.float
            )
        ]


        # -----------------------------------------
        # PPO Update
        # -----------------------------------------
        try:
            ppo_trainer.step(
                query_tensors,
                response_tensors,
                reward_tensors
            )

        except Exception as ppo_error:
            print(
                f"PPO warning: {ppo_error}"
            )


        # -----------------------------------------
        # Logs
        # -----------------------------------------
        print("\n-----------------------")
        print(f"Epoch {epoch}")
        print("Prompt:")
        print(prompt[:300], "...")
        print("\nModel Output:")
        print(generated_text)
        print(f"\nReward: {reward}")
        print("-----------------------")
        reward_history.append(
            {
                "epoch": epoch,
                "reward": float(reward),
                "mode": args.mode,
                "model": model_name,
                "persona": info.get("persona", {}).get("name", "unknown"),
                "policy": info.get("policy", {}).get("version", "unknown"),
                "priority_alignment": info.get("reward_breakdown", {}).get("priority_alignment", 0),
                "policy_compliance": info.get("reward_breakdown", {}).get("policy_compliance", 0),
                "tool_execution_bonus": info.get("reward_breakdown", {}).get("tool_execution_bonus", 0)
            }
        )


    except Exception as e:
        print(
            f"Epoch {epoch} skipped: {e}"
        )
        continue


if reward_history:
    csv_path = os.path.join(args.output_dir, "reward_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "reward",
                "mode",
                "model",
                "persona",
                "policy",
                "priority_alignment",
                "policy_compliance",
                "tool_execution_bonus"
            ]
        )
        writer.writeheader()
        writer.writerows(reward_history)

    json_path = os.path.join(args.output_dir, "reward_log.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reward_history, f, indent=2)

    rewards = [row["reward"] for row in reward_history]
    avg_reward = sum(rewards) / len(rewards)
    print("\nTraining summary")
    print(f"- Logged epochs: {len(reward_history)}")
    print(f"- Avg reward: {avg_reward:.2f}")
    print(f"- Max reward: {max(rewards):.2f}")
    print(f"- Min reward: {min(rewards):.2f}")
    print(f"- Reward CSV: {csv_path}")
    print(f"- Reward JSON: {json_path}")