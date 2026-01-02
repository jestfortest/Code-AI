#!/usr/bin/env python3
"""Fine-tune an instruction-following coding model with LoRA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


TEMPLATE_PREFIX = "<<SYS>>You are a concise senior software engineer who explains reasoning before sharing code.<<SYS>>"
PROMPT_TEMPLATE = "{prefix}\n\n### Prompt:\n{prompt}\n\n### Response:\n{response}\n"


def build_supervised_pairs(example: Dict[str, str]) -> Dict[str, str]:
    prompt_part = f"{TEMPLATE_PREFIX}\n\n### Prompt:\n{example['prompt']}\n\n### Response:\n"
    response_part = example["response"]
    return {"prompt_text": prompt_part, "response_text": response_part}


def detect_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    base_model: str
    dataset_path: Path
    output_dir: Path
    max_seq_length: int
    train: Dict[str, Any]
    lora: Dict[str, Any]

    @staticmethod
    def from_file(path: Path) -> "TrainConfig":
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        config_dir = path.parent.resolve()
        project_root = config_dir.parent

        def resolve_path(value: str) -> Path:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return candidate
            search_dirs = [config_dir, project_root]
            for idx, base in enumerate(search_dirs):
                resolved = (base / candidate).resolve()
                if resolved.exists() or idx == len(search_dirs) - 1:
                    return resolved
            return (project_root / candidate).resolve()

        return TrainConfig(
            base_model=data["base_model"],
            dataset_path=resolve_path(data["dataset_path"]),
            output_dir=resolve_path(data["output_dir"]),
            max_seq_length=int(data.get("max_seq_length", 512)),
            train=data.get("train", {}),
            lora=data.get("lora", {}),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train.yaml"),
        help="Path to YAML config describing the training run.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser


def format_dataset(dataset_path: Path) -> Dataset:
    ds = load_dataset("json", data_files=str(dataset_path))
    ds = ds["train"]

    ds = ds.map(
        build_supervised_pairs,
        remove_columns=[],
        desc="Formatting prompts",
    )
    return ds


def tokenize_dataset(
    ds: Dataset,
    tokenizer,
    max_seq_length: int,
    *,
    desc: str,
) -> Dataset:
    prompt_key = "prompt_text"
    response_key = "response_text"

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        # When batched=True, batch is a dict with lists of values
        inputs = batch.get(prompt_key, [])
        responses = batch.get(response_key, [])

        concatenated = [inp + resp for inp, resp in zip(inputs, responses)]
        result = tokenizer(
            concatenated,
            truncation=True,
            max_length=max_seq_length,
        )

        labels = []
        for input_ids, inp_text in zip(result["input_ids"], inputs):
            prompt_ids = tokenizer(
                inp_text,
                truncation=True,
                max_length=max_seq_length,
            )["input_ids"]
            label = [-100] * len(input_ids)
            prompt_len = min(len(prompt_ids), len(input_ids))
            for idx in range(prompt_len, len(input_ids)):
                label[idx] = input_ids[idx]
            labels.append(label)

        result["labels"] = labels
        return result

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=[prompt_key, response_key],
        desc=desc,
    )
    return tokenized


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainConfig.from_file(args.config)

    dataset_path = config.dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = format_dataset(dataset_path)
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
        config.max_seq_length,
        desc="Tokenizing train split",
    )
    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer,
        config.max_seq_length,
        desc="Tokenizing eval split",
    )

    device = detect_device()
    torch_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    def load_base_model():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        # Explicitly convert to fp16 if model came in as bf16
        if torch_dtype == torch.float16 and model.dtype != torch.float16:
            model = model.to(torch.float16)
        return model

    model = load_base_model()

    lora_cfg = LoraConfig(
        r=config.lora.get("r", 8),
        lora_alpha=config.lora.get("alpha", 16),
        lora_dropout=config.lora.get("dropout", 0.05),
        target_modules=config.lora.get(
            "target_modules",
            ["q_proj", "v_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if device.type != "cuda" or "device_map" not in model_kwargs:
        model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Enable fp16 for CUDA and MPS if requested in config
    fp16 = bool(config.train.get("fp16", False)) and device.type in {"cuda", "mps"}
    # Only enable bf16 if explicitly requested and on CUDA (bf16 not well-supported on MPS)
    bf16 = bool(config.train.get("bf16", False)) and device.type == "cuda"
    # Ensure fp16 and bf16 are mutually exclusive
    if fp16:
        bf16 = False

    training_kwargs = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.train.get("batch_size", 1),
        "gradient_accumulation_steps": config.train.get("gradient_accumulation", 1),
        "num_train_epochs": config.train.get("epochs", 3),
        "learning_rate": config.train.get("learning_rate", 2e-4),
        "warmup_ratio": config.train.get("warmup_ratio", 0.05),
        "logging_steps": config.train.get("logging_steps", 10),
        "save_steps": config.train.get("save_steps", 200),
        "evaluation_strategy": "steps",
        "eval_steps": config.train.get("eval_steps", 200),
        "fp16": fp16,
        "bf16": bf16,
        "report_to": "none",
        "save_total_limit": 2,
        "ddp_find_unused_parameters": False,
        # Disable pin_memory on MPS (not supported)
        "dataloader_pin_memory": device.type != "mps",
    }

    try:
        train_args = TrainingArguments(**training_kwargs)
    except TypeError as exc:
        if "evaluation_strategy" in str(exc):
            eval_steps = training_kwargs.pop("eval_steps", None)
            training_kwargs.pop("evaluation_strategy", None)
            if eval_steps is not None:
                training_kwargs["eval_strategy"] = "steps"
                training_kwargs["eval_steps"] = eval_steps
            train_args = TrainingArguments(**training_kwargs)
            # Suppress warning - we've already handled the compatibility
        else:
            raise

    # Use processing_class instead of tokenizer (transformers 5.0+ compatibility)
    # Try processing_class first, fall back to tokenizer for older versions
    try:
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
    except TypeError:
        # Fallback for older transformers versions that still use tokenizer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    trainer.train(resume_from_checkpoint=args.resume_from)

    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    adapter_path = config.output_dir / "adapter_model.pth"
    torch.save(model.state_dict(), adapter_path)

    metadata = {
        "base_model": config.base_model,
        "dataset": str(dataset_path),
        "max_seq_length": config.max_seq_length,
        "train_params": config.train,
        "lora": config.lora,
    }
    with (config.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Training complete. Adapter weights saved to {adapter_path}")


if __name__ == "__main__":
    main()
