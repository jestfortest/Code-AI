#!/usr/bin/env python3
"""Iteratively improve the dataset using Gemini feedback."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from src.gemini_eval import GeminiEvaluator
from src.model_interface import CodingModel, InferenceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/codellama-coder"),
        help="Directory containing adapter weights + metadata.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("training/data/coding_qa.jsonl"),
        help="Source dataset to sample prompts from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/coding_qa_augmented.jsonl"),
        help="Destination JSONL containing original + improved samples.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSONL log of Gemini reviews.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="How many prompts to review (sampled uniformly).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=80.0,
        help="Minimum Gemini score to accept without modification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling prompts.",
    )
    return parser


def load_examples(path: Path) -> List[Dict[str, Any]]:
    dataset = load_dataset("json", data_files=str(path))["train"]
    return dataset.to_list()


def call_with_retry(
    func,
    *args,
    retries: int = 3,
    delay: float = 2.0,
    backoff: float = 1.5,
    **kwargs,
):
    last_exc: Optional[Exception] = None
    current_delay = delay
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(current_delay)
            current_delay *= backoff
    if last_exc:
        raise last_exc
    raise RuntimeError("call_with_retry failed without exception")  # pragma: no cover


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    if not examples:
        raise ValueError(f"Dataset at {args.dataset} is empty")

    rng = random.Random(args.seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected = indices[: min(args.max_samples, len(indices))]

    cfg = InferenceConfig(run_dir=args.run_dir)
    model = CodingModel(cfg)
    evaluator = GeminiEvaluator.from_env()

    augmented_map: Dict[str, str] = {}
    logs: List[Dict[str, Any]] = []

    for idx in tqdm(selected, desc="Gemini reviewing", unit="sample"):
        example = examples[idx]
        prompt = example["prompt"]
        reference = example.get("response")

        try:
        answer = model.ask_question(prompt)
        except Exception as exc:
            logs.append({"prompt": prompt, "error": f"model_generation_failed: {exc}"})
            continue

        try:
            review = call_with_retry(
                evaluator.evaluate,
                prompt,
                answer,
                reference=reference,
            )
        except Exception as exc:
            logs.append(
                {
                    "prompt": prompt,
                    "model_answer": answer,
                    "error": f"review_failed: {exc}",
                }
            )
            continue

        raw_score = review.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        verdict = str(review.get("verdict", "")).upper()

        improved = None
        if score < args.score_threshold or verdict == "FAIL":
            feedback = review.get("feedback") or review.get("raw")
            try:
                improved = call_with_retry(
                    evaluator.suggest_improvement,
                prompt=prompt,
                candidate=answer,
                feedback=feedback,
                reference=reference,
            )
            except Exception as exc:
                logs.append(
                    {
                        "prompt": prompt,
                        "model_answer": answer,
                        "review": review,
                        "error": f"improvement_failed: {exc}",
                    }
                )
                improved = None

        final_response = improved or answer
        augmented_map[prompt] = final_response
        logs.append(
            {
                "prompt": prompt,
                "model_answer": answer,
                "review": review,
                "improved_response": improved,
            }
        )

    combined: List[Dict[str, str]] = []
    seen_prompts = set()
    for example in examples:
        prompt = example["prompt"]
        if prompt in augmented_map:
            combined.append({"prompt": prompt, "response": augmented_map[prompt]})
        else:
            combined.append(example)
        seen_prompts.add(prompt)
    for prompt, response in augmented_map.items():
        if prompt not in seen_prompts:
            combined.append({"prompt": prompt, "response": response})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in combined:
            json.dump(row, fh)
            fh.write("\n")

    log_path = args.log_path or args.output.with_suffix(".feedback.log.jsonl")
    with log_path.open("w", encoding="utf-8") as fh:
        for row in logs:
            json.dump(row, fh)
            fh.write("\n")

    print(
        f"Added {len(augmented)} reviewed samples to {args.output}. Logs saved to {log_path}."
    )


if __name__ == "__main__":
    main()
