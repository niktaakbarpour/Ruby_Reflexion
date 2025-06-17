import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load JSONL file
with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/edge_first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Explode tags so each tag gets its own row
df_exploded = df.explode('tags')
df_exploded['is_solved'] = df_exploded['is_solved'].astype(str)

# Count combinations of tag and is_solved
heatmap_data = df_exploded.groupby(['tags', 'is_solved']).size().unstack(fill_value=0)

# Sort tags by total number of problems
heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).index]

# Plot heatmap
plt.figure(figsize=(12, len(heatmap_data) * 0.25))  # Dynamic height
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
plt.title('Heatmap: Solved vs. Unsolved by Tag (All Tags)')
plt.xlabel('Is Solved')
plt.ylabel('Tag')
plt.tight_layout()
plt.show()
