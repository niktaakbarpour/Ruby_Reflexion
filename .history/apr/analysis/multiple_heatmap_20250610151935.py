import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Map each prompt to its JSONL file path
prompt_files = {
    "Reflexion": "C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/edge_first_edgeIO_CoTIO_CoT.jsonl",
    "SCOT": "C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/first_edgeIO_CoTIO_CoT.jsonl",
    # "HoarePrompt": "C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/hoareprompt.jsonl",
    # Add more prompts here if needed
}

# Collect all records
all_data = []

for prompt_name, file_path in prompt_files.items():
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {prompt_name}: {file_path}")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            tag = record.get('tag')
            is_solved = int(record.get('is_solved', 0))  # Default to 0 if missing
            
            if tag is not None:
                all_data.append({
                    'prompt': prompt_name,
                    'tag': tag,
                    'is_solved': is_solved
                })

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Group by tag and prompt, then compute solve rate
pivot_df = df.groupby(['tag', 'prompt'])['is_solved'].mean().unstack().fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    pivot_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={"label": "Proportion Solved"}
)

plt.title("Comparison of Prompting Techniques by Tag", fontsize=14)
plt.xlabel("Prompt", fontsize=12)
plt.ylabel("Tag", fontsize=12)
plt.xticks(fontsize=10, rotation=30)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
# plt.savefig("prompt_tag_heatmap.pdf", format='pdf')
