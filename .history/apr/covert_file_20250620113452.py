from datasets import Dataset

dataset = Dataset.from_file("data.arrow")
dataset.to_json("output.jsonl")
