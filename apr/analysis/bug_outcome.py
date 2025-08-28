import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Load data
jsonl_path = "../results/ds.jsonl"  # <-- update with your actual path
data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Define all verdict types
verdicts = [
    "COMPILATION ERROR", "MEMORY_LIMIT_EXCEEDED", "PASSED",
    "RUNTIME ERROR", "TIME_LIMIT_EXCEEDED", "WRONG ANSWER"
]

# Initialize counters
heatmap_counts = {
    (before, after): {"T": 0, "U": 0, "C": 0}
    for before in verdicts for after in verdicts
}

# Fill counts
for entry in data:
    before = entry.get("bug_exec_outcome", "UNKNOWN")
    after_verdicts = [v.get("verdict", "UNKNOWN") for v in entry.get("final_unit_test_results", [])]
    verdict_freq = defaultdict(int)
    for v in after_verdicts:
        verdict_freq[v] += 1

    # If there are verdicts after RAMP
    if verdict_freq:
        # Pick the most common one (mode)
        after = max(verdict_freq.items(), key=lambda x: x[1])[0]
    else:
        after = "UNKNOWN"

    # Skip unknowns
    if before not in verdicts or after not in verdicts:
        continue

    heatmap_counts[(before, after)]["T"] += 1
    if before == after:
        heatmap_counts[(before, after)]["U"] += 1
    else:
        heatmap_counts[(before, after)]["C"] += 1

# Convert to DataFrame for heatmap
labels = []
values = []
for before in verdicts:
    row_labels = []
    row_values = []
    for after in verdicts:
        cell = heatmap_counts[(before, after)]
        text = f"T: {cell['T']}\nU: {cell['U']}\nC: {cell['C']}"
        row_labels.append(text)
        row_values.append(cell["T"])
    labels.append(row_labels)
    values.append(row_values)

# Plot heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(values, annot=labels, fmt="", cmap="inferno", cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
ax.set_xticklabels(verdicts, rotation=45, ha='right')
ax.set_yticklabels(verdicts, rotation=0)
ax.set_xlabel("Exec Outcome After RAMP")
ax.set_ylabel("Exec Outcome Before RAMP")
plt.title("Heatmap of Bug Execution Transition After RAMP")
plt.tight_layout()
# plt.savefig("ramp_exec_transition_heatmap.png", dpi=300)
plt.show()
