from datasets import load_dataset
import pandas as pd

# Load the dataset
apr_dataset = load_dataset("NTU-NLP-sg/xCodeEval", "apr", trust_remote_code=True)

# Convert the train split to a DataFrame
df = apr_dataset["validation"].to_pandas()

# Display first few rows
print(df.head())
