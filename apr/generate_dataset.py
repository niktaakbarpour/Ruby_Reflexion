import sys
from datasets.load import load_dataset
import jsonlines
from typing import List

assert len(sys.argv) == 2, "Usage: python generate_dataset.py <MultiPL-E huggingface dataset name>"
DATASET_NAME = sys.argv[1]

def write_jsonl(path: str, data: List[dict], append: bool = False):
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def download_dataset(dataset_name: str):
    dataset = load_dataset("nuprl/MultiPL-E", dataset_name)

    final = []
    for item in dataset["test"]:
        name = item["name"]
        entry = "_".join(name.split("_")[2:])
        print(entry)
        item["entry_point"] = entry
        item["test"] = item["tests"]
        item["test"] = item["test"][1:] # there is some garbage at the start
        del item["tests"]
        final.append(item)

    output_path = f"./programming_runs/benchmarks/{dataset_name}.jsonl"
    _output_file = open(output_path, "w").close()

    
    write_jsonl(output_path, final)
    print(f"dumped to `{output_path}`")


if __name__ == "__main__":
    download_dataset(DATASET_NAME)
