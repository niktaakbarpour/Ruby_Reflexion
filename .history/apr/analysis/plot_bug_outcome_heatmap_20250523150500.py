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
flare_cmap = sns.color_palette("crest", as_cmap=True)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap=flare_cmap, linewidths=.5)
plt.title("Heatmap: Solved vs. Unsolved by Bug Execution Outcome")
plt.xlabel("Is Solved")
plt.ylabel("Bug Execution Outcome")
plt.tight_layout()
plt.show()
