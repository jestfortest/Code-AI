"""Sample driver that runs multiple coding prompts through the model and Gemini."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# #region agent log
def _agent_log(message: str, data: dict, hypothesis_id: str) -> None:
    payload = {
        "sessionId": "debug-session",
        "runId": "post-fix",
        "hypothesisId": hypothesis_id,
        "location": "src/run.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = Path("/Users/aidenchang/Documents/important/projects/Python/Cursor/.cursor/debug.log")
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# #endregion

try:
    from src.model_interface import CodingModel, InferenceConfig
except Exception as exc:  # pragma: no cover - import failure instrumentation
    _agent_log(
        "import_failed",
        {
            "cwd": os.getcwd(),
            "sys_path": sys.path,
            "project_root": str(PROJECT_ROOT),
            "error": repr(exc),
        },
        hypothesis_id="H1",
    )
    raise

PROMPTS = [
    # Python
    "Refactor a Python data-processing pipeline that recalculates expensive aggregates.",
    "Debug a Python memory leak caused by circular references.",
    "Rewrite a recursive Python function to be iterative for large inputs.",
    "Optimize a NumPy-based computation that is slower than expected.",
    "Design a Python plugin system using entry points.",
    "Explain Python GIL implications for CPU-bound vs I/O-bound workloads.",
    "Harden a Python CLI tool with proper argument validation and exit codes.",
    "Convert synchronous Python code to asyncio with proper cancellation handling.",
    "Profile a slow Python script and explain the bottlenecks.",
    "Design a Python retry mechanism with exponential backoff and jitter.",

    # TypeScript / JavaScript
    "Fix a TypeScript API client that mishandles 429 retry-after headers.",
    "Refactor a large JavaScript file into smaller ES modules.",
    "Debug a Node.js event-loop blocking issue.",
    "Migrate a JavaScript codebase from CommonJS to ES modules.",
    "Design a type-safe configuration loader in TypeScript.",
    "Fix a memory leak in a long-running Node.js service.",
    "Explain differences between debounce and throttle with examples.",
    "Secure a Node.js service against prototype pollution.",
    "Optimize a Webpack build that is excessively slow.",
    "Convert promise chains into async/await with proper error handling.",

    # Go
    "Design a Go interface for pluggable storage engines and provide a simple in-memory impl.",
    "Optimize Go code suffering from excessive allocations.",
    "Explain Go context propagation in HTTP services.",
    "Refactor Go code to avoid goroutine leaks.",
    "Design a Go worker pool with graceful shutdown.",
    "Debug a Go data race reported by the race detector.",
    "Explain when to use channels vs mutexes in Go.",
    "Implement retries with backoff in a Go HTTP client.",
    "Refactor Go error handling to add context.",
    "Design a Go REST API with proper layering.",

    # Rust
    "Optimize a Rust iterator pipeline that clones data on every step.",
    "Explain Rust ownership issues in async code.",
    "Refactor Rust code to eliminate unnecessary lifetimes.",
    "Design a safe Rust wrapper around an unsafe C API.",
    "Debug a Rust borrow checker error involving mutable references.",
    "Optimize Rust code by reducing heap allocations.",
    "Explain Send and Sync traits with practical examples.",
    "Design a Rust trait-based plugin architecture.",
    "Refactor Rust code to use iterators instead of loops.",
    "Explain interior mutability patterns in Rust.",

    # Web / Backend
    "Add unit tests for a Django view ensuring pagination metadata is correct.",
    "Outline steps to modularize a monolithic Flask blueprint into smaller packages.",
    "Show how to harden an Express.js endpoint against SQL injection and XSS.",
    "Design REST API pagination and filtering conventions.",
    "Debug a slow PostgreSQL query using EXPLAIN.",
    "Refactor a REST API to follow clean architecture principles.",
    "Design an authentication system using JWTs and refresh tokens.",
    "Explain tradeoffs between REST and GraphQL.",
    "Implement rate limiting in an API gateway.",
    "Design idempotent endpoints for payment processing.",

    # Frontend
    "Given a failing React hook, debug stale-closure issues and propose fixes.",
    "Optimize React rendering caused by unnecessary re-renders.",
    "Refactor a large React component into reusable hooks.",
    "Debug a CSS layout issue involving flexbox and overflow.",
    "Explain differences between controlled and uncontrolled components.",
    "Design a frontend state management strategy for a large app.",
    "Optimize bundle size in a React application.",
    "Explain how React reconciliation works.",
    "Debug a hydration mismatch in a Next.js app.",
    "Design accessible UI components following WCAG.",

    # Systems / Architecture
    "Explain how to migrate a Node.js callback-based service to async/await.",
    "Provide a clean-room description of how to detect race conditions in Kotlin coroutines.",
    "Design a message-driven system using Kafka.",
    "Explain eventual consistency with real-world examples.",
    "Design a caching strategy for a read-heavy service.",
    "Analyze a distributed system outage and propose mitigations.",
    "Explain CAP theorem tradeoffs in practice.",
    "Design a background job processing system.",
    "Compare monolith vs microservices architectures.",
    "Design a feature-flag system with safe rollouts.",

    # DevOps / Tooling
    "Design a CI pipeline for a multi-language repository.",
    "Debug a flaky test suite in CI.",
    "Explain Docker layer caching and optimization.",
    "Design Kubernetes liveness and readiness probes.",
    "Refactor infrastructure-as-code to reduce duplication.",
    "Explain blue-green vs canary deployments.",
    "Optimize container startup times.",
    "Design secrets management for cloud services.",
    "Explain observability pillars with examples.",
    "Design alerting rules that avoid noise.",
]


def main() -> None:
    cfg = InferenceConfig(run_dir=Path("outputs/codellama-coder"))
    model = CodingModel(cfg)

    for idx, prompt in enumerate(PROMPTS, start=1):
        print("=" * 80)
        print(f"[Prompt {idx}] {prompt}\n")
        answer = model.ask_question(prompt)
        print("Model answer:\n", answer, "\n")

        try:
            review = model.ask_with_gemini_review(prompt)
            print("Gemini review:\n", review["review"])
        except Exception as exc:  # pragma: no cover - optional
            print(f"[warning] Gemini review failed: {exc}")

    print("=" * 80)
    score = model.evaluate_exact_match(Path("training/data/coding_qa.jsonl"), limit=20)
    print("Quick dataset score:", score)


if __name__ == "__main__":
    main()