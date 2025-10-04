import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This script loads JSONL results, expands the tags field so each tag is counted separately, tallies solved vs. unsolved problems per tag, and plots the distribution as a grouped bar chart.

# Load your JSONL data
with open('', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Explode the tags list so each tag gets its own row
df_exploded = df.explode('tags')
df_exploded['is_solved'] = df_exploded['is_solved'].astype(str)

# Count solved/unsolved per tag
tag_counts = df_exploded.groupby(['tags', 'is_solved']).size().reset_index(name='count')

# Sort tags by total count (top to bottom)
total_counts = tag_counts.groupby('tags')['count'].sum().sort_values(ascending=False).index
df_sorted = tag_counts.copy()
df_sorted['tags'] = pd.Categorical(df_sorted['tags'], categories=total_counts, ordered=True)

# Plot with Seaborn
plt.figure(figsize=(14, 8))
sns.barplot(
    data=df_sorted,
    x='tags',
    y='count',
    hue='is_solved',
    palette={'False': '#cb4335', 'True': '#117a65'}
)

plt.ylabel('Number of Problems')
plt.xlabel('Tag')
plt.xticks(rotation=90)
plt.legend(title='is_solved')
plt.tight_layout()
# plt.savefig("tags_seaborn.png", format='png')
plt.show()
