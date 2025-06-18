import json
import pandas as pd
import matplotlib.pyplot as plt

# Load your JSONL data
with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/edge_first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Explode the tags list so each tag gets its own row
df_exploded = df.explode('tags')
df_exploded['is_solved'] = df_exploded['is_solved'].astype(str)

# Count solved/unsolved per tag
tag_counts = df_exploded.groupby(['tags', 'is_solved']).size().unstack(fill_value=0)

# Sort by total problems per tag
tag_counts = tag_counts.loc[tag_counts.sum(axis=1).sort_values(ascending=False).index]

# Plot grouped bar chart
ax = tag_counts.plot(kind='bar', figsize=(14, 8))
ax.set_title('Solved vs. Unsolved by Tag (All Tags)')
ax.set_ylabel('Number of Problems')
ax.set_xlabel('Tag')
plt.xticks(rotation=90)
ax.legend(title='is_solved')
plt.tight_layout()
plt.savefig("tags.pdf", format='pdf')
plt.show()