"""Utilities for loading, querying, and evaluating the fine-tuned coding model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_GEMINI_MODEL = "models/gemini-1.5-pro-latest"
DEFAULT_GEMINI_RUBRIC = (
    "Score the candidate response for clarity, correctness, and completeness. "
    "Return JSON with `score`, `verdict`, and `feedback`."
)

try:  # pragma: no cover - optional dependency
    from .gemini_eval import (
        GeminiEvaluator,
        DEFAULT_MODEL as _GEMINI_MODEL,
        DEFAULT_RUBRIC as _GEMINI_RUBRIC,
    )

    DEFAULT_GEMINI_MODEL = _GEMINI_MODEL
    DEFAULT_GEMINI_RUBRIC = _GEMINI_RUBRIC
except Exception:  # pragma: no cover
    GeminiEvaluator = None  # type: ignore

TEMPLATE_PREFIX = "<<SYS>>You are a concise senior software engineer who explains reasoning before sharing code.<<SYS>>"
PROMPT_TEMPLATE = "{prefix}\n\n### Prompt:\n{prompt}\n\n### Response:\n"


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
class InferenceConfig:
    run_dir: Path
    device: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

    @staticmethod
    def from_dir(run_dir: Path, **kwargs) -> "InferenceConfig":
        return InferenceConfig(run_dir=run_dir, **kwargs)


class CodingModel:
    """Convenience wrapper for chatting with the fine-tuned model."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.run_dir = config.run_dir
        self.metadata = self._load_metadata()

        base_model_name = self.metadata["base_model"]
        lora_cfg = self._build_lora_config(self.metadata["lora"])

        self.device = detect_device(config.device)
        torch_dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.run_dir if (self.run_dir / "tokenizer_config.json").exists() else base_model_name,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_kwargs = {}
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        def load_base_model():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=torch_dtype,
                    **model_kwargs,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch_dtype,
                    **model_kwargs,
                )
            # Explicitly convert to fp16 if model came in as bf16
            if torch_dtype == torch.float16 and model.dtype != torch.float16:
                model = model.to(torch.float16)
            return model

        model = load_base_model()
        model = get_peft_model(model, lora_cfg)
        if self.device.type != "cuda" or "device_map" not in model_kwargs:
            model.to(self.device)

        adapter_path = self.run_dir / "adapter_model.pth"
        if not adapter_path.is_file():
            raise FileNotFoundError(
                f"Expected adapter weights at {adapter_path}. Run training first."
            )

        state_dict = torch.load(adapter_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warning] Missing keys: {missing}, unexpected keys: {unexpected}")

        self.model = model.eval()

    def _load_metadata(self) -> Dict[str, str]:
        metadata_path = self.run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Could not find run metadata at {metadata_path}. Did training finish?"
            )
        with metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _build_lora_config(raw_cfg: Dict[str, Any]) -> LoraConfig:
        return LoraConfig(
            r=raw_cfg.get("r", 8),
            lora_alpha=raw_cfg.get("alpha", 16),
            lora_dropout=raw_cfg.get("dropout", 0.05),
            target_modules=raw_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

    def ask_question(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate an answer for a coding question."""

        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        )

        formatted = PROMPT_TEMPLATE.format(prefix=TEMPLATE_PREFIX, prompt=prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def evaluate_exact_match(self, dataset_path: Path, limit: int = 20) -> Dict[str, float]:
        """Run a lightweight exact-match evaluation on a subset of the dataset."""

        ds = load_dataset("json", data_files=str(dataset_path))["train"]
        limit = min(limit, len(ds))
        subset = ds.select(range(limit))

        def normalize(text: str) -> str:
            return " ".join(text.strip().lower().split())

        matches = 0
        generations = []
        for example in subset:
            prediction = self.ask_question(example["prompt"], temperature=0.0)
            expected = example["response"]
            generations.append({
                "prompt": example["prompt"],
                "prediction": prediction,
                "expected": expected,
            })
            if normalize(prediction) == normalize(expected):
                matches += 1

        score = matches / limit if limit else 0.0
        return {"exact_match": score, "samples": generations}

    def ask_with_gemini_review(
        self,
        prompt: str,
        *,
        reference: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        rubric: Optional[str] = None,
        **generation_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an answer and request a Gemini critique."""

        answer = self.ask_question(prompt, **generation_kwargs)

        if GeminiEvaluator is None:
            raise ImportError(
                "Gemini evaluation requires google-generativeai. "
                "Install it and ensure GEMINI_API_KEY is set."
            )

        evaluator = (
            GeminiEvaluator(
                api_key=gemini_api_key,
                model=gemini_model or DEFAULT_GEMINI_MODEL,
                rubric=rubric or DEFAULT_GEMINI_RUBRIC,
            )
            if gemini_api_key
            else GeminiEvaluator.from_env(
                model=gemini_model or DEFAULT_GEMINI_MODEL,
                rubric=rubric or DEFAULT_GEMINI_RUBRIC,
            )
        )

        review = evaluator.evaluate(prompt, answer, reference=reference)
        return {"answer": answer, "review": review}


__all__ = ["InferenceConfig", "CodingModel"]
