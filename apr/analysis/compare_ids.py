#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, Set, Tuple, Optional

# Compares two JSON/JSONL files by extracting bug_code_uid and apr_id (including from nested source_data) and prints which IDs are unique to each file.
def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yields records from a file that is either:
      - a JSON array of objects, or
      - JSONL (one JSON object per line), or
      - a single JSON object.
    """
    # Try parse as whole JSON first
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec
        elif isinstance(obj, dict):
            # If it contains a likely list container, use that
            for key in ("data", "records", "items", "results"):
                if isinstance(obj.get(key), list):
                    for rec in obj[key]:
                        if isinstance(rec, dict):
                            yield rec
                    break
            else:
                # Single object
                yield obj
        return
    except json.JSONDecodeError:
        pass  # fall back to JSONL

    # Fallback: JSON Lines
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec

def _maybe_parse_json_str(x: Any) -> Any:
    if isinstance(x, str) and x.lstrip().startswith("{"):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def extract_ids(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (bug_code_uid, apr_id) from a record.
    Looks under source_data first, then top-level.
    """
    sd = rec.get("source_data", {})
    sd = _maybe_parse_json_str(sd)
    bug = apr = None

    if isinstance(sd, dict):
        bug = sd.get("bug_code_uid") or sd.get("bug_uid") or sd.get("bug_id")
        apr = sd.get("apr_id")

    # Fallbacks to top-level if not found
    bug = bug or rec.get("bug_code_uid") or rec.get("bug_uid") or rec.get("bug_id")
    apr = apr or rec.get("apr_id")

    # Normalize to strings
    bug = str(bug) if bug is not None else None
    apr = str(apr) if apr is not None else None
    return bug, apr

def collect_sets(path: Path) -> Tuple[Set[str], Set[str]]:
    bug_set: Set[str] = set()
    apr_set: Set[str] = set()
    for rec in iter_records(path):
        bug, apr = extract_ids(rec)
        if bug:
            bug_set.add(bug)
        if apr:
            apr_set.add(apr)
    return bug_set, apr_set

def print_diff(title: str, only_a: Set[str], only_b: Set[str], name_a: str, name_b: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Only in {name_a} (count {len(only_a)}):")
    if only_a:
        for x in sorted(only_a):
            print("  ", x)
    else:
        print("  (none)")
    print(f"Only in {name_b} (count {len(only_b)}):")
    if only_b:
        for x in sorted(only_b):
            print("  ", x)
    else:
        print("  (none)")

def main():
    ap = argparse.ArgumentParser(description="Compare bug_code_uid and apr_id across two JSON/JSONL files.")
    ap.add_argument("first", type=Path, help="Path to first JSON/JSONL file")
    ap.add_argument("second", type=Path, help="Path to second JSON/JSONL file")
    args = ap.parse_args()

    f1, f2 = args.first, args.second
    bug1, apr1 = collect_sets(f1)
    bug2, apr2 = collect_sets(f2)

    print(f"Loaded:\n  {f1} -> bug_code_uid:{len(bug1)}  apr_id:{len(apr1)}"
          f"\n  {f2} -> bug_code_uid:{len(bug2)}  apr_id:{len(apr2)}")

    # Differences
    bug_only_1 = bug1 - bug2
    bug_only_2 = bug2 - bug1
    apr_only_1 = apr1 - apr2
    apr_only_2 = apr2 - apr1

    print_diff("bug_code_uid differences", bug_only_1, bug_only_2, f1.name, f2.name)
    print_diff("apr_id differences", apr_only_1, apr_only_2, f1.name, f2.name)

if __name__ == "__main__":
    main()
