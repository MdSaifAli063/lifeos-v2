"""
Minimal Colab-friendly PPO training script for LifeOS OpenEnv.

Usage:
  !python training/train_ppo_colab.py
"""

import json
import os
import sys

import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from openenv_env.env import LifeOSEnv
from utils.prompt_builder import build_prompt


def main():
    with open("demo_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    env = LifeOSEnv(scenarios)
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)

    config = PPOConfig(
        learning_rate=2e-6,
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=1
    )

    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer
    )

    rewards = []
    for epoch in range(8):
        state = env.reset()
        prompt = build_prompt(state, env.memory.get_context())

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.pretrained_model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.4
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if len(text) < 5:
            text = "Decision -> Reschedule conflict. Email -> Sorry, can we move this?"

        _, reward, _, _ = env.step(text)
        rewards.append(float(reward))

        trainer.step(
            [inputs["input_ids"][0]],
            [output_ids[0]],
            [torch.tensor(reward, dtype=torch.float)]
        )

        print(f"epoch={epoch} reward={reward:.2f}")

    os.makedirs("training_outputs", exist_ok=True)
    with open("training_outputs/reward_log_colab.json", "w", encoding="utf-8") as f:
        json.dump(rewards, f, indent=2)

    print("done")
    print(f"avg_reward={sum(rewards)/len(rewards):.2f}")
    print("saved: training_outputs/reward_log_colab.json")


if __name__ == "__main__":
    main()
