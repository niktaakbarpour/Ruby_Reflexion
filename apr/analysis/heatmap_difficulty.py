#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Builds a difficulty-vs-after-outcome heatmap from JSON/JSONL by extracting difficulty (with optional mapping/binning) and unit-test exec outcomes, aggregating counts, and saving an annotated heatmap image.

# “After” outcomes stay the same
AFTER_CATEGORIES = [
    "COMPILATION_ERROR",
    "MEMORY_LIMIT_EXCEEDED",
    "PASSED",
    "RUNTIME_ERROR",
    "TIME_LIMIT_EXCEEDED",
    "WRONG_ANSWER",
]

def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yields dict records from:
      - JSONL file (one JSON per line),
      - JSON file containing a dict (single record) or a list of dicts (many records).
    """
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            for rec in data:
                yield rec
        else:
            raise ValueError("Unsupported JSON structure at top level.")


def _get_by_dotpath(d: Dict[str, Any], dotpath: str) -> Any:
    """
    Safely extract a nested value using dot notation, e.g. "source_data.difficulty".
    Returns None if any level is missing.
    """
    cur = d
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _extract_difficulty(
    record: Dict[str, Any],
    field: str,
    diff_map: Optional[Dict[str, str]] = None,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Extract and normalize difficulty label for one record.

    - field: dot-path to difficulty (e.g., "difficulty" or "source_data.difficulty")
    - diff_map: optional mapping from raw values to labels (keys coerced to str)
    - bins + labels: if provided and value is numeric, bin it: bins define edges (right-open),
      labels length must be len(bins)-1.
    """
    raw = _get_by_dotpath(record, field)
    if raw is None:
        return None

    # If a map is provided, apply it first (treat raw as string key).
    if diff_map is not None:
        key = str(raw)
        return diff_map.get(key, None)

    # If bins are provided, try numeric binning.
    if bins is not None and labels is not None:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        # np.digitize returns index in 1..len(bins)-1 if within range
        idx = np.digitize([val], bins, right=False)[0] - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return None

    # Otherwise, treat as categorical string
    return str(raw)


def _extract_pairs_for_record(
    record: Dict[str, Any],
    diff_field: str,
    diff_map: Optional[Dict[str, str]],
    bins: Optional[List[float]],
    labels: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """
    From one problem record, produce pairs:
      (difficulty_label, unit_test.exec_outcome) for every unit test.
    """
    diff_label = _extract_difficulty(record, diff_field, diff_map, bins, labels)
    if diff_label is None:
        return []

    utr = record.get("unit_test_results", [])
    tests: List[Dict[str, Any]] = []
    if isinstance(utr, list):
        if len(utr) == 1 and isinstance(utr[0], list):
            tests = [t for t in utr[0] if isinstance(t, dict)]
        else:
            for item in utr:
                if isinstance(item, dict) and "exec_outcome" in item:
                    tests.append(item)
                elif isinstance(item, list):
                    tests.extend([t for t in item if isinstance(t, dict)])

    pairs: List[Tuple[str, str]] = []
    for t in tests:
        aft = t.get("exec_outcome")
        if aft is not None:
            pairs.append((diff_label, aft))
    return pairs


def build_matrix(
    pairs: List[Tuple[str, str]],
    diff_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a Difficulty (rows) x AfterOutcome (cols) count matrix.
    If diff_order is provided, use it to order the difficulty rows.
    """
    df = pd.DataFrame(pairs, columns=["Difficulty", "After"])
    # Normalize “After” categories and ensure all columns exist
    df["After"] = pd.Categorical(df["After"], categories=AFTER_CATEGORIES, ordered=True)

    # Difficulty ordering
    if diff_order:
        df["Difficulty"] = pd.Categorical(df["Difficulty"], categories=diff_order, ordered=True)

    counts = (
        df.value_counts(["Difficulty", "After"])
          .rename("count")
          .reset_index()
          .pivot(index="Difficulty", columns="After", values="count")
          .reindex(columns=AFTER_CATEGORIES)
          .fillna(0)
          .astype(int)
    )

    # If diff_order was not provided, keep natural (sorted) order
    if "Difficulty" in df and not diff_order:
        # Reindex rows by appearance order of difficulty in the data (stable order)
        seen = list(dict.fromkeys(df["Difficulty"].astype(str).tolist()))
        counts = counts.reindex(index=seen)
    return counts


def plot_heatmap(counts: pd.DataFrame, out_path: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(counts.values, aspect="auto")

    # Axis labels and ticks
    ax.set_xlabel("Exec Outcome After RAMP")
    ax.set_ylabel("Difficulty")
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns, rotation=40, ha="right")
    ax.set_yticklabels(counts.index)

    # Annotate cells with counts
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = counts.iat[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center")

    if title:
        ax.set_title(title)

    # Colorbar (matplotlib default colors per your rule)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved heatmap to {out_path}")


def _parse_bins_and_labels(bin_str: Optional[str], label_str: Optional[str]) -> Tuple[Optional[List[float]], Optional[List[str]]]:
    if not bin_str and not label_str:
        return None, None
    if not bin_str or not label_str:
        raise SystemExit("--difficulty-bins and --difficulty-labels must be provided together.")
    try:
        bins = [float(x) for x in bin_str.split(",")]
    except ValueError:
        raise SystemExit("Failed to parse --difficulty-bins (must be comma-separated numbers).")
    labels = [s.strip() for s in label_str.split(",")]
    if len(labels) != len(bins) - 1:
        raise SystemExit("Number of --difficulty-labels must be len(bins)-1.")
    return bins, labels


def main():
    p = argparse.ArgumentParser(description="Build Difficulty vs After-Outcome heatmap.")
    p.add_argument("--input", required=True, help="Path to .jsonl or .json")
    p.add_argument("--output", default="difficulty_exec_outcome_heatmap.png", help="Output image path (PNG)")
    p.add_argument("--title", default="", help="Optional plot title")

    # Difficulty extraction controls
    p.add_argument("--difficulty-field", default="difficulty",
                   help="Dot-path to difficulty (e.g., 'difficulty' or 'source_data.difficulty').")
    p.add_argument("--difficulty-map", default=None,
                   help="JSON mapping from raw values to labels (e.g., '{\"1\":\"EASY\",\"2\":\"MEDIUM\",\"3\":\"HARD\"}').")
    p.add_argument("--difficulty-bins", default=None,
                   help="Comma-separated numeric bin edges (e.g., '0,0.33,0.66,1'). Requires --difficulty-labels.")
    p.add_argument("--difficulty-labels", default=None,
                   help="Comma-separated labels for bins (len = len(bins)-1).")
    p.add_argument("--difficulty-order", default=None,
                   help="Comma-separated desired row order (e.g., 'EASY,MEDIUM,HARD').")

    args = p.parse_args()

    # Parse optional settings
    diff_map = json.loads(args.difficulty_map) if args.difficulty_map else None
    bins, labels = _parse_bins_and_labels(args.difficulty_bins, args.difficulty_labels)
    diff_order = [s.strip() for s in args.difficulty_order.split(",")] if args.difficulty_order else None

    in_path = Path(args.input)
    pairs_all: List[Tuple[str, str]] = []
    for rec in _iter_records(in_path):
        pairs_all.extend(_extract_pairs_for_record(rec, args.difficulty_field, diff_map, bins, labels))

    if not pairs_all:
        raise SystemExit("No (difficulty, test) pairs found. Check input structure or difficulty settings.")

    counts = build_matrix(pairs_all, diff_order)
    plot_heatmap(counts, Path(args.output), args.title)


if __name__ == "__main__":
    main()
