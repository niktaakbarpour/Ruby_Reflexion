import json
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
jsonl_path = "../results/ds.jsonl"  # Change as needed
bin_width = 250
min_difficulty = 750
max_difficulty = 2500

# --- Read data ---
records = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        difficulty = entry.get("difficulty")
        pass_value = entry.get("pass@1")
        if difficulty is None or pass_value is None:
            continue
        records.append({
            "difficulty": difficulty,
            "pass@1": int(pass_value == 1.0)
        })

df = pd.DataFrame(records)

# --- Bin difficulties ---
bins = pd.interval_range(start=min_difficulty, end=max_difficulty + bin_width, freq=bin_width, closed='left')
df["difficulty_bin"] = pd.cut(df["difficulty"], bins=bins)

# --- Count pass/fail per bin ---
df["status"] = df["pass@1"].map({1: "Pass", 0: "Fail"})
grouped = df.groupby(["difficulty_bin", "status"]).size().unstack(fill_value=0)

# --- Plot ---
ax = grouped.plot(kind="bar", figsize=(10, 6), color=["salmon", "seagreen"])
# plt.title("Pass@1 vs Difficulty")
plt.xlabel("Difficulty")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle=":", alpha=0.7)
plt.legend(title="Pass@1")
plt.tight_layout()
plt.savefig("pass1_vs_difficulty.png", dpi=300)
# plt.show()
