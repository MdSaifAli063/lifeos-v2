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
# Load scenarios
# -------------------------------------------------
with open("demo_scenarios.json", "r", encoding="utf-8") as f:
    scenarios = json.load(f)

env = LifeOSEnv(scenarios)


# -------------------------------------------------
# Model
# -------------------------------------------------
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    model_name
)

ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    model_name
)


# -------------------------------------------------
# Stable PPO Config
# -------------------------------------------------
config = PPOConfig(
    learning_rate=1e-6,
    batch_size=1,
    mini_batch_size=1,
    ppo_epochs=2
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
for epoch in range(50):

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
            max_new_tokens=60,
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
        _, reward, _, _ = env.step(
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


    except Exception as e:
        print(
            f"Epoch {epoch} skipped: {e}"
        )
        continue