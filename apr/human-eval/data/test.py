import gzip
import os

file_path=os.path.dirname(os.path.abspath(__file__))+"\HumanEval.jsonl.gz"
with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for i in range(10):  # Print the first 10 lines
        print(f.readline().strip())
