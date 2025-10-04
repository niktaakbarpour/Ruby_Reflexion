import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Loads JSONL results for multiple prompting techniques, computes the proportion of solved cases per bug_exec_outcome for each prompt, and visualizes the comparison as a heatmap.

# Map each prompt to its JSONL file path
prompt_files = {
    "Reflexion": "",
    "SCOT": "",
}

records = []

for prompt_name, file_path in prompt_files.items():
    if not os.path.exists(file_path):
        print(f"Missing file for {prompt_name}: {file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            outcome = d.get('bug_exec_outcome')
            is_solved = int(d.get('is_solved', 0))

            if outcome is not None:
                records.append({
                    'prompt': prompt_name,
                    'bug_exec_outcome': outcome,
                    'is_solved': is_solved
                })

# Convert to DataFrame
df = pd.DataFrame(records)

# Group by outcome + prompt, calculate % solved
pivot_df = df.groupby(['bug_exec_outcome', 'prompt'])['is_solved'].mean().unstack().fillna(0)

# Plot heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(
    pivot_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={"label": "Proportion Solved"}
)

plt.title("Prompting Technique Comparison by Bug Execution Outcome", fontsize=14)
plt.xlabel("Prompt", fontsize=12)
plt.ylabel("Bug Execution Outcome", fontsize=12)
plt.xticks(fontsize=10, rotation=30)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()