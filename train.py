import datasets
import logging
import os
import re
import sys
import torch
import transformers

from dataclasses import dataclass, field
from datasets import Dataset
from datetime import datetime
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

from rewards import (
    PAD_TOKEN,
    format_reward_func,
    subset_correct_reward_func,
    token_count_reward_func,
)


@dataclass
class PTConfig:
    base_dir: str = (
        field(
            default=".",
        ),
    )
    chat_template: str = field(
        default="""{%- if messages and messages[0]['role'] == 'system' -%}
    {%- set conversation = messages -%}
{%- else -%}
    {%- set conversation = [{'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'}] + (messages if messages else []) -%}
{%- endif -%}
{%- for message in conversation -%}
    {%- if message.role == 'system' -%}
        {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' -}}
    {%- elif message.role == 'user' -%}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' -}}
    {%- elif message.role == 'assistant' -%}
        {{- '<|im_start|>assistant\n' + (message.content | default('', true)) + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
{%- endif -%}"""
    )
    dataset_name: str = field(default="./data/subset_sum_l5-15_s2-10_n10000")
    dataset_subset: str = field(default=None)
    name: str = field(
        default="forescore",
    )
    system_prompt: str = field(
        default="""You are an expert at solving subset sum problems. When given a list of numbers and a target sum, you need to find which subset of those numbers adds up to exactly the target.

Think through the problem step-by-step:
1. Understand the available numbers and the target sum
2. Consider which numbers might combine to reach the target
3. Verify that your chosen subset sums to exactly the target
4. Present your answer as a list of the numbers in the subset

Here's an example showing the input format and required response:

Input:
Given the list: [3, 7, 12, 25, 45, 70]

Find a subset that sums to 82

What subset of numbers sums to exactly 82?

Response:
<think>
I need to find numbers from [3, 7, 12, 25, 45, 70] that sum to 82.

Let me work through this systematically:
- The largest number is 70, so I likely need that: 70 leaves 12 remaining
- From the remaining numbers [3, 7, 12, 25, 45], I need a subset summing to 12
- I can use 12 directly: 70 + 12 = 82

Let me verify: 70 + 12 = 82 ✓

The subset is [12, 70]
</think>
<answer>[12, 70]</answer>

Your response MUST follow this XML format:
<think>
[Your step-by-step reasoning here - show your work finding the subset]
</think>
<answer>
[The subset that sums to the target, e.g., [3, 7, 12] or in comma-separated format: 3, 7, 12]
</answer>

Be thorough in your thinking process, showing how you determined which numbers to include.
"""
    )


def make_conv_for_grpo(example, system_prompt):
    """
    Prepare subset sum examples for GRPO training.

    Expected example fields from dataset:
    - problem: Subset sum problem statement
    - numbers: List of available numbers
    - target: Target sum to find
    - solution: One valid subset that sums to target (for reference)
    """
    problem = example["problem"]
    numbers = example["numbers"]
    target = example["target"]

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
        "numbers": numbers,
        "target": target,
    }


def main():
    parser = TrlParser((PTConfig, GRPOConfig, ModelConfig))
    pt_args, training_args, model_args = parser.parse_args_and_config()

    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"
    os.environ["WANDB_PROJECT"] = "forescore"

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Post training parameters {pt_args}")
    logger.info(f"Training parameters {training_args}")

    # Set up output paths
    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    training_args.logging_dir = f"{pt_args.base_dir}/{pt_args.name}/logs"
    training_args.output_dir = f"{pt_args.base_dir}/{pt_args.name}/checkpoints"
    training_args.run_name = f"{pt_args.name}_{formatted_datetime}"

    # Load and preprocess dataset (tokenization is handled by GRPO Trainer)
    if not os.path.exists(pt_args.dataset_name):
        logger.error(f"Dataset not found at: {pt_args.dataset_name}")
        logger.error("Please generate the dataset first using generate.py")
        sys.exit(1)

    logger.info(f"Loading dataset from local path: {pt_args.dataset_name}")
    train_dataset = Dataset.load_from_disk(pt_args.dataset_name)

    # Preprocess dataset
    train_dataset = train_dataset.map(
        make_conv_for_grpo, fn_kwargs={"system_prompt": pt_args.system_prompt}
    )
    train_dataset.save_to_disk(f"{pt_args.base_dir}/{pt_args.name}/data")

    # Initialize the model
    model = AutoModelForCausalLM.from_pretrained(
        # model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation=model_args.attn_implementation,
        # use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # tokenizer.pad_token = PAD_TOKEN
    tokenizer.chat_template = pt_args.chat_template

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        bias="none",
    )

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            subset_correct_reward_func,
            token_count_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # Training and Evaluation
    logger.info(f"\nStarting training for {training_args.num_train_epochs} epochs.")

    # Check for last checkpoint
    ckpt = None
    if training_args.resume_from_checkpoint is not None:
        ckpt = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        ckpt = get_last_checkpoint(training_args.output_dir)
        if ckpt:
            logger.info(f"\nCheckpoint detected, resuming training at {ckpt=}.")
        else:
            logger.info("\nNo checkpoint detected, starting training from scratch.")

    try:
        train_result = trainer.train(resume_from_checkpoint=ckpt)
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
    finally:
        del trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
