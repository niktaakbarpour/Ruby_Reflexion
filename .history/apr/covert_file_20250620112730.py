from datasets import load_dataset
import pandas as pd

# Load only the validation split
validation_dataset = load_dataset(
    "NTU-NLP-sg/xCodeEval",
    "apr",
    split="validation",
    trust_remote_code=True
)

# Convert to pandas DataFrame
df = validation_dataset.to_pandas()

# Display first few rows
print(df.head())
