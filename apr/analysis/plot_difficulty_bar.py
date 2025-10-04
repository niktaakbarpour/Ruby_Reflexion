import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Plots solved vs. unsolved counts by difficulty bin from a JSONL dataset and overlays a line showing the total number of problems per bin.

# Load data
with open('', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Define difficulty bins
bins = [0, 1200, 1400, 1600, 1800, 2000, 3000]
labels = ['<1200', '1200–1400', '1400–1600', '1600–1800', '1800–2000', '2000+']
df['difficulty_bin'] = pd.cut(df['difficulty'], bins=bins, labels=labels, right=False)

# Count solved vs unsolved per bin
bar_data = df.groupby(['difficulty_bin', 'is_solved']).size().unstack(fill_value=0)

# Compute total per bin
total_counts = df.groupby('difficulty_bin').size()

# Choose hex colors for bars: one for 'False' (unsolved), one for 'True' (solved)
custom_colors = ['#cb4335', '#117a65']

# Plot grouped bar chart
ax = bar_data.plot(kind='bar', figsize=(10, 6), color=custom_colors, alpha=0.8)

# Overlay line chart for total questions per difficulty
line = ax.plot(range(len(total_counts)), total_counts.values,
               marker='o', linestyle='-', color='#3c5c8f', linewidth=2, label='Total')

# --- Add labels above line chart dots ---
for x, y in enumerate(total_counts.values):
    ax.annotate(f'{int(y)}',
                (x+0.1, y-0.1),
                ha='center', va='bottom', fontsize=12, xytext=(0, 4), textcoords='offset points')

# Labeling
plt.xlabel('Difficulty Range')
plt.ylabel('Number of Problems')

# Force integer ticks
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.legend(title='', labels=['Total number of questions', 'Solved', 'Total'])
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
# plt.savefig("difficulty_with_total.png", format='png')
plt.show()

