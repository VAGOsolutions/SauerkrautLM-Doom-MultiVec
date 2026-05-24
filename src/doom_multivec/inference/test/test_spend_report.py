"""Simple spend summary for local LLM usage logs.

Reads `src/doom_multivec/inference/usage/usage_log.txt` (or a custom
path from `USAGE_LOG_PATH`) and prints a compact summary of spend,
tokens, and calls. This avoids needing the admin spend-report endpoint.

Run:
    python test_spend_report.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _parse_timestamp(value: str) -> datetime | None:
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_log_entries(log_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not log_path.exists():
        return entries

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.6f}"


def _build_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "calls": len(entries),
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "models": defaultdict(lambda: {"calls": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        "first_timestamp": None,
        "last_timestamp": None,
    }

    timestamps: List[datetime] = []

    for entry in entries:
        cost = entry.get("cost_usd")
        if isinstance(cost, (int, float)):
            summary["total_cost"] += float(cost)

        prompt_tokens = int(entry.get("prompt_tokens") or 0)
        completion_tokens = int(entry.get("completion_tokens") or 0)
        total_tokens = int(entry.get("total_tokens") or (prompt_tokens + completion_tokens))

        summary["total_prompt_tokens"] += prompt_tokens
        summary["total_completion_tokens"] += completion_tokens
        summary["total_tokens"] += total_tokens

        model = str(entry.get("model") or "unknown")
        model_bucket = summary["models"][model]
        model_bucket["calls"] += 1
        model_bucket["prompt_tokens"] += prompt_tokens
        model_bucket["completion_tokens"] += completion_tokens
        model_bucket["total_tokens"] += total_tokens
        if isinstance(cost, (int, float)):
            model_bucket["cost"] += float(cost)

        timestamp = entry.get("timestamp")
        if isinstance(timestamp, str):
            parsed = _parse_timestamp(timestamp)
            if parsed is not None:
                timestamps.append(parsed)

    if timestamps:
        summary["first_timestamp"] = min(timestamps).isoformat().replace("+00:00", "Z")
        summary["last_timestamp"] = max(timestamps).isoformat().replace("+00:00", "Z")

    return summary


def main() -> None:
    default_log_path = Path(__file__).resolve().parents[1] / "usage" / "usage_log.txt"
    log_path = Path(os.getenv("USAGE_LOG_PATH", str(default_log_path)))

    entries = _load_log_entries(log_path)
    if not entries:
        print(f"No usage log entries found at: {log_path}")
        return

    summary = _build_summary(entries)

    print(f"Usage log: {log_path}")
    print(f"Calls: {summary['calls']}")
    print(f"Total spend: {_format_money(summary['total_cost'])}")
    print(f"Total prompt tokens: {summary['total_prompt_tokens']}")
    print(f"Total completion tokens: {summary['total_completion_tokens']}")
    print(f"Total tokens: {summary['total_tokens']}")

    if summary["first_timestamp"] and summary["last_timestamp"]:
        print(f"Window: {summary['first_timestamp']} -> {summary['last_timestamp']}")

    print("\nBy model:")
    models = summary["models"]
    for model_name in sorted(models):
        model_summary = models[model_name]
        print(
            f"- {model_name}: calls={model_summary['calls']}, spend={_format_money(model_summary['cost'])}, "
            f"prompt_tokens={model_summary['prompt_tokens']}, completion_tokens={model_summary['completion_tokens']}, "
            f"total_tokens={model_summary['total_tokens']}"
        )

    print("\nLast 5 entries:")
    for entry in entries[-5:]:
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
