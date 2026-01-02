"""Gemini-based evaluator that scores generated coding answers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import google.generativeai as genai

DEFAULT_MODEL = "models/gemini-1.5-pro-latest"
DEFAULT_RUBRIC = (
    "Score the candidate code review on clarity, correctness, and completeness. "
    "Provide a JSON object with fields: `score` (0-100), `verdict` (PASS/FAIL), "
    "and `feedback` (actionable summary)."
)


def _extract_json_block(text: str) -> str:
    """Return the JSON substring even if Gemini wrapped it in fences."""

    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("json"):
                candidate = part[4:].strip()
                if candidate:
                    return candidate
            if part.startswith("{") and part.endswith("}"):
                return part
    return text


@dataclass
class GeminiEvaluator:
    api_key: str
    model: str = DEFAULT_MODEL
    rubric: str = DEFAULT_RUBRIC

    @classmethod
    def from_env(
        cls,
        env_var: str = "GEMINI_API_KEY",
        **kwargs: Any,
    ) -> "GeminiEvaluator":
        api_key = os.getenv(env_var)
        if not api_key:
            raise EnvironmentError(
                f"Missing {env_var}. Export your Gemini API key first."
            )
        return cls(api_key=api_key, **kwargs)

    def __post_init__(self) -> None:
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def evaluate(
        self,
        prompt: str,
        candidate: str,
        reference: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask Gemini to critique a generated answer."""

        instructions = (
            "You are an impartial senior engineer reviewing a model's output. "
            "Respond with JSON only."
        )
        payload = (
            f"{instructions}\n\nRubric: {self.rubric}\n\n"
            f"Task prompt:\n{prompt}\n\n"
            f"Candidate answer:\n{candidate}\n"
        )
        if reference:
            payload += f"\nReference answer (if any):\n{reference}\n"
        if extra_context:
            payload += f"\nExtra context:\n{extra_context}\n"

        response = self.client.generate_content(payload)
        text = response.text or ""
        cleaned = _extract_json_block(text)

        result: Dict[str, Any] = {"raw": text.strip()}
        try:
            result.update(json.loads(cleaned))
        except json.JSONDecodeError:
            result["parse_error"] = "Could not parse JSON output"
        return result

    def suggest_improvement(
        self,
        prompt: str,
        candidate: str,
        feedback: Optional[str] = None,
        reference: Optional[str] = None,
        style_hint: str = "Return a concise explanation followed by corrected code if needed.",
    ) -> str:
        """Ask Gemini to rewrite the candidate answer using the given feedback."""

        payload = (
            "You are a senior engineer iterating on a model's response after review. "
            "Return only the improved answer (no JSON)."
            f"\n\nOriginal task:\n{prompt}\n"
            f"\nCandidate answer:\n{candidate}\n"
        )
        if feedback:
            payload += f"\nFeedback to address:\n{feedback}\n"
        if reference:
            payload += f"\nReference answer (optional):\n{reference}\n"
        payload += f"\nGuidance:\n{style_hint}\n"

        response = self.client.generate_content(payload)
        return (response.text or "").strip()


__all__ = ["GeminiEvaluator", "DEFAULT_MODEL", "DEFAULT_RUBRIC"]
