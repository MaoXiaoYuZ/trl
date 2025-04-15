# PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5s PYTHONPATH=/hy-tmp/trl:$PYTHONPATH python -m debugpy --wait-for-client --listen 5678  examples/scripts/ppo/ppo_tldr.py \
PYTHONPATH=/hy-tmp/trl:$PYTHONPATH python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100