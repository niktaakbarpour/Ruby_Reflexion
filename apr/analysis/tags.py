#!/usr/bin/env python3
import json
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Reads a JSONL, computes per-tag solved percentages and totals, plots a horizontal bar chart with a secondary count axis, and prints the top tags by solved rate.

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def count_by_tag(jsonl_path):
    """
    Returns:
      solved_per_tag: Counter of solved counts per tag.
      total_per_tag: Counter of total counts per tag.
    """
    solved_per_tag = Counter()
    total_per_tag = Counter()

    for rec in stream_jsonl(jsonl_path):
        tags = rec.get("tags") or []
        if not isinstance(tags, (list, tuple)):
            continue
        tag_set = {str(t).strip() for t in tags if str(t).strip()}
        if not tag_set:
            continue

        is_solved = bool(rec.get("is_solved"))
        for tag in tag_set:
            total_per_tag[tag] += 1
            if is_solved:
                solved_per_tag[tag] += 1

    return solved_per_tag, total_per_tag

def plot_percentage_with_counts(solved_per_tag, total_per_tag, top_n=None, save_path=None):
    # Compute percentages
    percentages = {
        tag: (100.0 * solved_per_tag[tag] / total_per_tag[tag]) if total_per_tag[tag] else 0.0
        for tag in total_per_tag
    }

    # Sort tags by percentage desc (keeps your current behavior)
    items = sorted(percentages.items(), key=lambda x: (-x[1], x[0]))
    if top_n:
        items = items[:top_n]
    tags, pcts = zip(*items) if items else ([], [])

    # Counts in the same (sorted) tag order
    counts = [total_per_tag[tag] for tag in tags]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(tags))))
    y_pos = range(len(tags))

    # Barh for solved %
    bars = ax.barh(list(y_pos), pcts, color="#a8324a", alpha=0.8, label="Solved (%)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tags)
    ax.set_xlabel("Solved (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.7)

    # Headroom so 'implementation' label at w+12 isn't clipped
    max_pct = max(pcts) if pcts else 0
    extra = 15 if any(t.lower() == "implementation" for t in tags) else 8
    ax.set_xlim(0, max(100, max_pct + extra))

    # === Annotate each bar with its percentage (with special cases) ===
    for bar, pct, tag in zip(bars, pcts, tags):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        label = f"{pct:.1f}%"
        t = tag.lower()

        if t == "implementation":
            # place at w + 12 (outside, right)
            ax.text(w + 7, y, label, va="center", ha="left",
                    fontsize=11, color="#333333", clip_on=False)
            
        elif t == "brute force":
            # place at w - 3 (inside, near the end)
            x_pos = max(w + 7, 0.5)  # clamp just in case
            ax.text(x_pos, y, label, va="center", ha="right",
                    fontsize=11, color="#333333")
            
        elif t == "constructive algorithms":
            # place at w - 3 (inside, near the end)
            x_pos = max(w + 7, 0.5)  # clamp just in case
            ax.text(x_pos, y, label, va="center", ha="right",
                    fontsize=11, color="#333333")

        elif t == "math":
            # place at w - 3 (inside, near the end)
            x_pos = max(w - 3, 0.5)  # clamp just in case
            ax.text(x_pos, y, label, va="center", ha="right",
                    fontsize=11, color="white")

        else:
            # default behavior: inside if wide enough, otherwise just outside
            if w >= 12:
                ax.text(w - 1, y, label, va="center", ha="right",
                        fontsize=11, color="white")
            else:
                ax.text(w + 0.25, y, label, va="center", ha="left",
                        fontsize=9, color="#333333", clip_on=False)
    # ==================================================================

    # Secondary x-axis for counts
    ax2 = ax.twiny()
    ax2.plot(counts, list(y_pos), marker="o", color="#101178", linestyle="-",
             linewidth=3, markersize=7, label="Total questions")
    ax2.set_xlabel("Number of questions")

    ax.xaxis.set_major_locator(MultipleLocator(10))   # grid every 10% on solved %
    ax2.xaxis.set_major_locator(MultipleLocator(20))  # ticks for counts

    # Combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.8, 0), ncol=2)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Show percentage of solved questions per tag from a JSONL file.")
    ap.add_argument("--input", default="", help="Path to JSONL file (e.g., Ruby.jsonl)")
    ap.add_argument("--top-n", type=int, default=None, help="Show only top N tags by solved percentage")
    ap.add_argument("--save", default="tags.png", help="Path to save the percentage chart (PNG). If omitted, shows interactively.")
    # ap.add_argument("--save", help="Path to save the percentage chart (PNG). If omitted, shows interactively.")
    args = ap.parse_args()

    solved_per_tag, total_per_tag = count_by_tag(args.input)

    if not solved_per_tag:
        print("No tags or solved data found.")
        return

    # Chart: percentage bars with counts line on the same plot
    plot_percentage_with_counts(
        solved_per_tag,
        total_per_tag,
        top_n=args.top_n,
        save_path=args.save
    )

    # Print textual summary
    percentages = {
        tag: (100.0 * solved_per_tag[tag] / total_per_tag[tag]) if total_per_tag[tag] else 0.0
        for tag in total_per_tag
    }
    items = sorted(percentages.items(), key=lambda x: (-x[1], x[0]))
    head = items[: (args.top_n or 10)]
    print("\nTop tags by solved %:")
    for tag, pct in head:
        print(f"- {tag}: {pct:.1f}% ({solved_per_tag[tag]}/{total_per_tag[tag]})")

if __name__ == "__main__":
    main()
