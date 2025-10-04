#!/usr/bin/env python3
import argparse
from datasets import load_from_disk, DatasetDict

# This script loads a Hugging Face dataset from disk, filters records where lang == "Ruby", extracts their apr_id values, and either prints them to stdout or writes them to a text file.

def main(dataset_dir: str, out_path: str | None):
    obj = load_from_disk(dataset_dir)

    def iter_splits():
        if isinstance(obj, DatasetDict):
            for name, dset in obj.items():
                yield name, dset
        else:
            yield "data", obj

    found = []
    for split_name, dset in iter_splits():
        ruby = dset.filter(lambda ex: isinstance(ex.get("lang"), str) and ex["lang"].lower() == "ruby")
        if "apr_id" not in ruby.column_names:
            raise KeyError(f"'apr_id' column not found in split {split_name}. Columns: {ruby.column_names}")
        found.extend(ruby["apr_id"])

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for x in found:
                f.write(f"{x}\n")
    else:
        print("\n".join(map(str, found)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="Path to the dataset folder (e.g., filtered_10_percent_dataset)")
    ap.add_argument("-o", "--out", help="Optional output text file to write apr_ids")
    args = ap.parse_args()
    main(args.dataset_dir, args.out)
