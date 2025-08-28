#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_records(path):
    """Stream JSONL records."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(input_path, save_path=None, show=True):
    total_counts = Counter()
    anypass_counts = Counter()
    unit_iter_keys = None

    for rec in load_records(input_path):
        outcome = rec.get("bug_exec_outcome") or "UNKNOWN"

        if unit_iter_keys is None:
            unit_iter_keys = [k for k in rec.keys() if k.startswith("pass@1_unit_iter")]
            if not unit_iter_keys:
                unit_iter_keys = [f"pass@1_unit_iter{i}" for i in range(11)]

        total_counts[outcome] += 1

        def is_one(v):
            try:
                return float(v) == 1.0
            except Exception:
                return False

        any_success = any(is_one(rec.get(k, 0)) for k in unit_iter_keys)
        if any_success:
            anypass_counts[outcome] += 1

    # --- Compute percentages instead of raw counts ---
    outcomes = sorted(total_counts.keys())
    perc_list = [(anypass_counts[o] / total_counts[o]) * 100 if total_counts[o] > 0 else 0
                 for o in outcomes]

    # ----------------- Bullet Chart -----------------
    plt.figure(figsize=(8, 0.5 * len(outcomes)))
    y_positions = np.arange(len(outcomes))

    plt.barh(y_positions, perc_list, color="teal")
    plt.yticks(y_positions, outcomes)
    plt.xlim(0, 100)
    plt.xlabel("Pass@1")
    # plt.title("Percentage with ≥1 unit pass by bug_exec_outcome")

    # Annotate each bar with its percentage
    for i, v in enumerate(perc_list):
        plt.text(v + 1, i, f"{v:.1f}%", va="center")

    plt.tight_layout()
    # ------------------------------------------------

    # if save_path:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bullet chart of percentages where any pass@1_unit_iter{i} == 1.0, grouped by bug_exec_outcome."
    )
    parser.add_argument("--input", default="c:/Users/niktakbr/Desktop/Ruby_Reflexion/apr/results/ds.jsonl", help="Path to JSONL file")
    parser.add_argument("--save", default="c:/Users/niktakbr/Desktop/Ruby_Reflexion/apr/analysis/pasa1_over_outcomes.png", help="Optional path to save the figure (e.g., out.png)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure window")
    args = parser.parse_args()

    main(args.input, save_path=args.save, show=not args.no_show)
