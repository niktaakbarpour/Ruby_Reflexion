import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/edge_first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Define difficulty bins
bins = [0, 1200, 1400, 1600, 1800, 2000, 3000]
labels = ['<1200', '1200–1400', '1400–1600', '1600–1800', '1800–2000', '2000+']
df['difficulty_bin'] = pd.cut(df['difficulty'], bins=bins, labels=labels, right=False)

# Count solved vs unsolved per bin
bar_data = df.groupby(['difficulty_bin', 'is_solved']).size().unstack(fill_value=0)

# Plot grouped bar chart
bar_data.plot(kind='bar', figsize=(10, 6))
plt.title('Solved vs. Unsolved Problems per Difficulty Range')
plt.xlabel('Difficulty Range')
plt.ylabel('Number of Problems')
plt.legend(['Unsolved', 'Solved'], title='is_solved', labels=['False', 'True'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
