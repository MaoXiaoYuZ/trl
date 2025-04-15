# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train", num_proc=10)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
training_args.per_device_train_batch_size = 4
training_args.gradient_checkpointing = True
training_args.num_generations = 2
trainer = GRPOTrainer(
    # model="Qwen/Qwen2-0.5B-Instruct",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()