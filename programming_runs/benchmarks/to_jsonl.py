import pandas as pd
import sys

def parquet_to_jsonl(input_file, output_file):
    """Convert a Parquet file to JSONL format."""
    df = pd.read_parquet(input_file, engine="pyarrow")  # or "fastparquet"
    df.to_json(output_file, orient='records', lines=True)
    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.parquet> <output.jsonl>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    parquet_to_jsonl(input_path, output_path)