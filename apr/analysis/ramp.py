#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# This script loads samples from a JSON/JSONL file, extracts their IDs and final unit test pass flags, prints all sample IDs, and identifies which ones passed all final unit tests.

def load_samples(path: Path) -> List[Dict[str, Any]]:
    """
    Load multiple samples from a single file.
    Supports:
      - JSONL (one JSON object per line)
      - JSON array: [ {...}, {...}, ... ]
      - JSON object containing a list under common keys (samples/data/records/items/results)
      - Single JSON object (treated as one-sample list)
    """
    text = path.read_text(encoding="utf-8").strip()

    # Try whole-file JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("samples", "data", "records", "items", "results"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            return [obj]  # single object fallback
    except json.JSONDecodeError:
        pass

    # Fallback: JSON Lines
    samples: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                samples.append(obj)
        except json.JSONDecodeError:
            continue

    if not samples:
        raise ValueError(f"No valid JSON samples found in {path}")
    return samples


def extract_final_pass_flags(sample: Dict[str, Any]) -> List[bool]:
    """
    Collect all boolean `passed` values that live under any `final_unit_test_results`
    subtree in the sample.
    """
    passes: List[bool] = []

    def collect(node: Any):
        if isinstance(node, dict):
            # Record a boolean 'passed' flag if present
            if "passed" in node and isinstance(node["passed"], bool):
                passes.append(node["passed"])
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for x in node:
                collect(x)

    def walk(node: Any):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "final_unit_test_results":
                    collect(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(sample)
    return passes


def extract_id(sample: Dict[str, Any]) -> str | None:
    """Prefer apr_id, with sensible fallbacks."""
    sd = sample.get("source_data") or {}
    return (
        sample.get("apr_id")
        or sd.get("apr_id")
        or sample.get("task_id")
        # or sample.get("src_uid")
        or sample.get("id")
    )


def main(file_path: str) -> int:
    samples = load_samples(Path(file_path))
    print(len(samples))
    for sample in samples:
        apr_id_total = extract_id(sample)
        print(apr_id_total)
        pass_flags = extract_final_pass_flags(sample)
        # require at least one test and all must be True
        if pass_flags and all(pass_flags):
            _id = extract_id(sample)
            # if _id:
                # print(_id)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final_pass_ids.py <path-to-json-or-jsonl>", file=sys.stderr)
        sys.exit(2)
    file_path = " ".join(sys.argv[1:])  # tolerate spaces in the path without quoting
    sys.exit(main(file_path))
