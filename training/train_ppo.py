from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
import torch
import json

from openenv_env.env import LifeOSEnv
from utils.prompt_builder import build_prompt

# Load scenarios
with open("demo_scenarios.json") as f:
    scenarios = json.load(f)

env = LifeOSEnv(scenarios)

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

for epoch in range(50):
    state = env.reset()
    memory = env.memory.get_context()

    prompt = build_prompt(state, memory)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Keep only generated part
    generated_text = response[len(prompt):]

    _, reward, _, _ = env.step(generated_text)

    # PPO step
    ppo_trainer.step(
        [prompt],
        [generated_text],
        [reward]
    )

    print(f"Epoch {epoch} | Reward: {reward}")