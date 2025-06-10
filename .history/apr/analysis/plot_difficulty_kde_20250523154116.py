import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your JSON file
with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Load into a DataFrame
df = pd.DataFrame(data)

# Plot KDE histogram: difficulty distribution by is_solved
plt.figure(figsize=(10, 6))
sns.histplot(df[df['is_solved']]['difficulty'], color='blue', label='Solved', kde=True, stat="density", bins=30)
sns.histplot(df[~df['is_solved']]['difficulty'], color='red', label='Unsolved', kde=True, stat="density", bins=30)
plt.title('Difficulty Distribution: Solved vs. Unsolved')
plt.xlabel('Difficulty')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig("kde.pdf", format='pdf')

# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load your JSONL dataset
# with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f]

# df = pd.DataFrame(data)

# # Plot histogram with raw counts
# plt.figure(figsize=(12, 6))
# sns.histplot(df[df['is_solved']]['difficulty'], color='blue', label='Solved', kde=False, stat="count", bins=30)
# sns.histplot(df[~df['is_solved']]['difficulty'], color='red', label='Unsolved', kde=False, stat="count", bins=30)

# plt.title('Difficulty Distribution: Solved vs. Unsolved (Raw Counts)')
# plt.xlabel('Difficulty')
# plt.ylabel('Number of Problems')
# plt.legend()
# plt.tight_layout()
# plt.show()
