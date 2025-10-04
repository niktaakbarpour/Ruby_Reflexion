#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Generates an annotated heatmap of unit-test execution outcomes by difficulty from JSON/JSONL input (supporting optional difficulty mapping/binning).

CATEGORIES = [
    "COMPILATION_ERROR",
    "MEMORY_LIMIT_EXCEEDED",
    "PASSED",
    "RUNTIME_ERROR",
    "TIME_LIMIT_EXCEEDED",
    "WRONG_ANSWER",
]

def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    print("hello")
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

def _extract_pairs(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    From one problem record, produce pairs:
      (bug_exec_outcome, unit_test.exec_outcome) for every unit test.
    We replicate the single bug outcome once per unit test.
    """
    bug = (record.get("source_data") or {}).get("bug_exec_outcome")
    if bug is None:
        return []

    utr = record.get("unit_test_results", [])
    tests: List[Dict[str, Any]] = []
    if isinstance(utr, list):
        # Common observed shape: [ [ {exec_outcome: ...}, ... ] ]
        if len(utr) == 1 and isinstance(utr[0], list):
            tests = [t for t in utr[0] if isinstance(t, dict)]
        else:
            # Fallback: flatten any lists of dicts one level deep
            for item in utr:
                if isinstance(item, dict) and "exec_outcome" in item:
                    tests.append(item)
                elif isinstance(item, list):
                    tests.extend([t for t in item if isinstance(t, dict)])

    pairs = []
    for t in tests:
        aft = t.get("exec_outcome")
        if aft is not None:
            pairs.append((bug, aft))
    return pairs

def build_matrix(pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(pairs, columns=["Before", "After"])
    # Ensure categories are ordered and all combinations present
    df["Before"] = pd.Categorical(df["Before"], categories=CATEGORIES, ordered=True)
    df["After"]  = pd.Categorical(df["After"],  categories=CATEGORIES, ordered=True)
    counts = (
        df.value_counts(["Before", "After"])
          .rename("count")
          .reset_index()
          .pivot(index="Before", columns="After", values="count")
          .reindex(index=CATEGORIES, columns=CATEGORIES)
          .fillna(0)
          .astype(int)
    )
    return counts

def plot_heatmap(
    counts: pd.DataFrame,
    out_path: Path,
    title: str = "",
    cmap: str = "YlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    # Optional normalization bounds
    norm = None
    if vmin is not None or vmax is not None:
        vmin_eff = np.min(counts.values) if vmin is None else vmin
        vmax_eff = np.max(counts.values) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=vmin_eff, vmax=vmax_eff)

    alpha: int = 0.8
    im = ax.imshow(counts.values, aspect="auto", cmap=cmap, norm=norm, alpha=alpha)

    # Axis labels and ticks
    ax.set_xlabel("Exec Outcome After RAMP")
    ax.set_ylabel("Exec Outcome Before RAMP")
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns, rotation=40, ha="right")
    ax.set_yticklabels(counts.index)


    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = counts.iat[i, j]
            # condition: row "PASSED", col "WRONG_ANSWER"
            if counts.index[i] == "WRONG_ANSWER" and counts.columns[j] == "PASSED":
                color = "white"
            else:
                color = "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12)



    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved heatmap to {out_path}")

    if show:
        plt.show()
    plt.close(fig)  # avoid state/caching issues

def main():
    p = argparse.ArgumentParser(description="Build bug-vs-test outcome heatmap.")
    p.add_argument("--input", required=True, help="Path to .jsonl or .json")
    p.add_argument("--output", default="exec_outcome_heatmap2.png", help="Output image path (PNG)")
    p.add_argument("--title", default="", help="Optional plot title")
    p.add_argument("--cmap", default="YlGn", help="Matplotlib colormap (e.g., viridis, magma, Blues, YlGnBu)")
    p.add_argument("--vmin", type=float, default=None, help="Optional lower bound for color normalization")
    p.add_argument("--vmax", type=float, default=None, help="Optional upper bound for color normalization")
    p.add_argument("--show", action="store_true", help="Show the plot window after saving")
    args = p.parse_args()

    in_path = Path(args.input)
    pairs_all: List[Tuple[str, str]] = []
    for rec in _iter_records(in_path):
        pairs_all.extend(_extract_pairs(rec))

    if not pairs_all:
        raise SystemExit("No (bug, test) pairs found. Check input structure.")

    counts = build_matrix(pairs_all)
    plot_heatmap(
        counts,
        Path(args.output),
        title=args.title,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        show=args.show,
    )

if __name__ == "__main__":
    main()
