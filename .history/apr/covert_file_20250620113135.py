import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

# Load the Arrow file
with open('data.arrow', 'rb') as f:
    reader = ipc.RecordBatchFileReader(f)
    table = reader.read_all()

# Convert Arrow table to pandas DataFrame
df = table.to_pandas()

# Save as JSON Lines
df.to_json('output.jsonl', orient='records', lines=True)
