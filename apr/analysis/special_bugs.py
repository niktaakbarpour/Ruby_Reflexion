#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Iterable

# Reads a JSON/JSONL file, prints the total and each sample’s ID (with fallbacks), collects all unit-test exec_outcome values, and identifies samples where every test PASSED.

def load_samples(path: Path) -> list[dict]:
    """
    Load multiple samples from a single file.
    Supports:
      - JSONL (one JSON object per line)
      - JSON array: [ {...}, {...}, ... ]
      - JSON object containing a list under common keys (samples/data/records/items/results)
      - Single JSON object (treated as one-sample list)
    """
    text = path.read_text(encoding="utf-8").strip()

    # Try as whole-file JSON first
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
    samples: list[dict] = []
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


def extract_exec_outcomes(sample: dict) -> list[str]:
    """Recursively collect all exec_outcome values under unit_test_results."""
    outcomes: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if "exec_outcome" in node:
                outcomes.append(str(node["exec_outcome"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(sample.get("unit_test_results", []))
    return outcomes


def extract_apr_id(sample: dict) -> str | None:
    sd = sample.get("source_data") or {}
    return (
        sd.get("apr_id")
        or sample.get("apr_id")
        or sd.get("aprId")
        or sample.get("aprId")
        or sample.get("task_id")   # <-- added
        or sample.get("src_uid")   # <-- added
        or sample.get("id")        # <-- common fallback
    )



def main(file_path: str) -> int:
    path = Path(file_path)
    samples = load_samples(path)
    print(len(samples))
    for sample in samples:
        apr_id_total = extract_apr_id(sample)
        print(apr_id_total)
        outcomes = extract_exec_outcomes(sample)
        # Only accept if there is at least one test and all passed
        if outcomes and all(o.upper() == "PASSED" for o in outcomes):
            apr_id = extract_apr_id(sample)
            # if apr_id:
                # print(apr_id)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python single_file_apr_ids.py <path-to-json-or-jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
