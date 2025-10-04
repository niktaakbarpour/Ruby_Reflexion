import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creates a boxplot of problem difficulty (Ruby only), categorizing each case as Initially Fixed, Fixed by RAMP, or Remain Unfixed based on pass@1_unit_iter results across iterations.

# --- Config ---
jsonl_path = ""  # Change as needed
num_iters = 11

# --- Collect labeled difficulty data ---
data = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        difficulty = entry.get("difficulty")
        if difficulty is None:
            continue

        # Round pass@1_unit_iter values to integer
        unit_passes = [round(entry.get(f"pass@1_unit_iter{i}", 0)) for i in range(num_iters)]

        if unit_passes[0] == 1:
            label = "Initially Fixed"
        elif any(unit_passes[1:]):
            label = "Fixed by RAMP"
        else:
            label = "Remain Unfixed"

        data.append({"difficulty": difficulty, "status": label, "language": "Ruby"})

# --- Convert to DataFrame ---
df = pd.DataFrame(data)

# --- Plot ---
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="language",
    y="difficulty",
    hue="status",
    palette={
        "Initially Fixed": "steelblue",
        "Fixed by RAMP": "mediumseagreen",
        "Remain Unfixed": "indianred"
    },
    hue_order=["Initially Fixed", "Fixed by RAMP", "Remain Unfixed"]
)

plt.xlabel("Programming Language")
plt.ylabel("Difficulty")
# plt.title("Repair Success by Difficulty (Ruby Only)")
plt.legend(title="", loc="upper left")
plt.tight_layout()
# plt.savefig("ramp_boxplot_ruby_only.png", dpi=300)
plt.show()
