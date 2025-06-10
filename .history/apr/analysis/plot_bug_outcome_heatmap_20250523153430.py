import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your JSONL file
with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure is_solved is a string for readable heatmap labels
df['is_solved'] = df['is_solved'].astype(str)

# Compute the counts
heatmap_data = df.groupby(['bug_exec_outcome', 'is_solved']).size().unstack(fill_value=0)

# Create flare colormap
flare_cmap = sns.color_palette("flare", as_cmap=True)

# Plot heatmap
plt.figure(figsize=(6, 4))  # smaller figure => smaller cells
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap=flare_cmap,
    linewidths=0.5,
    annot_kws={"size": 12},  # larger numbers in cells
    cbar_kws={"shrink": 0.8}
)

plt.xlabel("Is Solved", fontsize=14)
plt.ylabel("Bug Execution Outcome", fontsize=14)
plt.title("Heatmap: Solved vs. Unsolved by Bug Execution Outcome", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
