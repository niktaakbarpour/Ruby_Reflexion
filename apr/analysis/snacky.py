import json
import pandas as pd
import matplotlib.pyplot as plt

# Computes per-iteration counts of passed vs. failed samples from a JSONL dataset and visualizes their progression over iterations as a stacked area chart with a boundary line.

# Load data
jsonl_path = ""
records = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

total_samples = len(records)

# Build per-iteration stats
results = {"iteration": [], "Correct": [], "Incorrect": []}

# Iter 0: all unsolved
results["iteration"].append(0)
results["Correct"].append(0)
results["Incorrect"].append(total_samples)

# Iter 1–11
for i in range(11):
    correct = sum(1 for r in records if r.get(f"pass@1_unit_iter{i}", 0) == 1)
    incorrect = total_samples - correct
    results["iteration"].append(i+1)
    results["Correct"].append(correct)
    results["Incorrect"].append(incorrect)

print(results)

# DataFrame for plotting
df = pd.DataFrame(results)

# Colors
correct_color = "#66c2a5"     # soft green
incorrect_color = "#fc7d92"   # soft red
boundary_color = "#333333"    # dark gray line

# Prepare plot
plt.figure(figsize=(10, 6))
plt.stackplot(df["iteration"], df["Correct"], df["Incorrect"],
              colors=[correct_color, incorrect_color], alpha=0.7)

# Boundary line
plt.plot(df["iteration"], df["Correct"], color=boundary_color, linewidth=4.0)

# Labels
mid_iter = df["iteration"].iloc[len(df)//2]
mid_correct = df["Correct"].iloc[len(df)//2] // 2
mid_incorrect = df["Correct"].iloc[len(df)//2] + df["Incorrect"].iloc[len(df)//2] // 2

plt.text(mid_iter, mid_correct, "Passed", color="black", fontsize=15,
         ha="center", va="center")
plt.text(mid_iter, mid_incorrect, "Failed", color="black", fontsize=15,
         ha="center", va="center")

plt.xlabel("Iteration")
plt.ylabel("Number of Samples")
# plt.grid(alpha=0.3)

# Axes
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xticks(df["iteration"])  # show all ticks

ax = plt.gca()
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# plt.savefig("new_snacky", dpi=200, bbox_inches="tight")
plt.show()
