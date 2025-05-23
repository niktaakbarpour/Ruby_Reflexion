import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your JSON file
with open('../results/first_edgeIO_CoTIO_CoT.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Create histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df[df['is_solved'] == True]['difficulty'], color='blue', label='Solved', kde=True, stat="density", bins=30)
sns.histplot(df[df['is_solved'] == False]['difficulty'], color='red', label='Unsolved', kde=True, stat="density", bins=30)

plt.title('Difficulty Distribution: Solved vs. Unsolved')
plt.xlabel('Difficulty')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()
