#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Iterable, Dict, Any, Union

# This script streams records from a JSON/JSONL file, checks whether all unit tests passed (exec_outcome == "PASSED"), and prints the apr_id (with optional fields like src_id or bug_uid) for fully passing samples.

def stream_records(path: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    """
    Yields sample objects from either:
      - JSONL (one JSON object per line), or
      - a JSON array file, or
      - a single JSON object file.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        s = head.lstrip()

        if s.startswith("["):
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        elif s.startswith("{") and s.rstrip().endswith("}"):
            obj = json.load(f)
            if isinstance(obj, dict):
                yield obj
        else:
            # Assume JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def flatten_unit_tests(node: Any) -> Iterable[Dict[str, Any]]:
    """Flattens arbitrarily nested lists/dicts under unit_test_results."""
    if isinstance(node, dict):
        yield node
    elif isinstance(node, list):
        for x in node:
            yield from flatten_unit_tests(x)

def all_tests_passed(sample: Dict[str, Any]) -> bool:
    """
    Returns True iff every unit test result has exec_outcome == 'PASSED'.
    (If there are zero results, returns False.)
    """
    units = list(flatten_unit_tests(sample.get("unit_test_results", [])))
    if not units:
        return False
    return all(u.get("exec_outcome") == "PASSED" for u in units)

def main(in_path: str):
    for sample in stream_records(in_path):
        if all_tests_passed(sample):
            sd = sample.get("source_data", {}) or {}
            out = sd.get("apr_id") or sample.get("apr_id"),
                # map requested names to fields in your data
                # "src_id": sd.get("src_uid") or sd.get("src_id") or sample.get("src_id"),
                # "bug_uid": sd.get("bug_code_uid") or sd.get("bug_uid") or sample.get("bug_uid"),
            
            print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print apr_id, src_id, bug_uid for samples where all exec_outcome == PASSED."
    )
    parser.add_argument("input", help="Path to JSON/JSONL file")
    args = parser.parse_args()
    main(args.input)
