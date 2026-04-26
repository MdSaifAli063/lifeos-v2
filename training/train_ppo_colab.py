"""
Minimal Colab-friendly PPO training script for AetherMind OpenEnv.

Usage in Colab:
  !python training/train_ppo_colab.py --epochs 8 --mode fast
"""

import argparse
import csv
import json
import os
import sys

import torch
from transformers import AutoTokenizer
try:
    from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
except Exception as trl_import_error:
    raise ImportError(
        "PPO APIs not found in installed TRL version. "
        "Install a PPO-compatible version in Colab:\n"
        "!pip install -q 'trl==0.8.6'"
    ) from trl_import_error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openenv_env.env import LifeOSEnv
from utils.prompt_builder import build_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Colab-friendly PPO training runner")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument(
        "--mode",
        choices=["fast", "default"],
        default="fast",
        help="fast vs default only changes schedule; both use google/flan-t5-large (app default)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "training_outputs"),
        help="Where to write reward logs",
    )
    return parser.parse_args()


def get_mode_config(mode):
    if mode == "default":
        return {
            "model_name": "google/flan-t5-large",
            "learning_rate": 1e-6,
            "ppo_epochs": 2,
            "max_new_tokens": 60,
        }
    return {
        "model_name": "google/flan-t5-large",
        "learning_rate": 2e-6,
        "ppo_epochs": 1,
        "max_new_tokens": 40,
    }


def main():
    args = parse_args()
    cfg = get_mode_config(args.mode)

    scenarios_path = os.path.join(PROJECT_ROOT, "demo_scenarios.json")
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    env = LifeOSEnv(scenarios)
    model_name = cfg["model_name"]
    print(f"[colab-train] mode={args.mode} model={model_name} epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)

    trainer = PPOTrainer(
        config=PPOConfig(
            learning_rate=cfg["learning_rate"],
            batch_size=1,
            mini_batch_size=1,
            ppo_epochs=cfg["ppo_epochs"],
        ),
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    logs = []
    for epoch in range(args.epochs):
        state = env.reset()
        prompt = build_prompt(state, env.memory.get_context())

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.pretrained_model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.4,
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if len(text) < 5:
            text = "Decision -> Reschedule conflict. Email -> Sorry, can we move this?"

        _, reward, _, info = env.step(text)
        trainer.step(
            [inputs["input_ids"][0]],
            [output_ids[0]],
            [torch.tensor(reward, dtype=torch.float)],
        )

        row = {
            "epoch": epoch,
            "reward": float(reward),
            "mode": args.mode,
            "model": model_name,
            "persona": info.get("persona", {}).get("name", "unknown"),
            "policy": info.get("policy", {}).get("version", "unknown"),
        }
        logs.append(row)
        print(f"[colab-train] epoch={epoch:02d} reward={reward:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "reward_log_colab.json")
    csv_path = os.path.join(args.output_dir, "reward_log_colab.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "reward", "mode", "model", "persona", "policy"])
        writer.writeheader()
        writer.writerows(logs)

    rewards = [r["reward"] for r in logs] or [0.0]
    print("[colab-train] done")
    print(f"[colab-train] avg_reward={sum(rewards)/len(rewards):.2f}")
    print(f"[colab-train] saved_json={json_path}")
    print(f"[colab-train] saved_csv={csv_path}")


if __name__ == "__main__":
    main()
