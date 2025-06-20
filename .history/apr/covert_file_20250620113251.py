import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Try reading it as a Parquet file
table = pq.read_table("data.arrow")
df = table.to_pandas()

# Convert to JSONL
df.to_json("output.jsonl", orient="records", lines=True)
