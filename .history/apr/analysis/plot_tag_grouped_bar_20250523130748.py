import json
import pandas as pd
import matplotlib.pyplot as plt

# Load your JSONL data
with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Explode tags into individual rows
df_exploded = df.explode('tags')

# Convert is_solved to string for clearer labeling
df_exploded['is_solved'] = df_exploded['is_solved'].astype(str)

# Count frequency of each tag
tag_totals = df_exploded['tags'].value_counts()

# Select top 15 most frequent tags
top_tags = tag_totals.head(15).index
df_top = df_exploded[df_exploded['tags'].isin(top_tags)]

# Count solved/unsolved per tag
tag_counts = df_top.groupby(['tags', 'is_solved']).size().unstack(fill_value=0)

# Plot grouped bar chart
ax = tag_counts.plot(kind='bar', figsize=(12, 6))
ax.set_title('Solved vs. Unsolved by Tag (Top 15 Tags)')
ax.set_ylabel('Number of Problems')
ax.set_xlabel('Tag')
plt.xticks(rotation=45, ha='right')
ax.legend(title='is_solved')
plt.tight_layout()
plt.show()
