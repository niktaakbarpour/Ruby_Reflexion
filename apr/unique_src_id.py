# import json

# def count_unique_src_ids(jsonl_file):
#     unique_src_ids = set()  # A set to store unique src_ids
#     with open(jsonl_file, 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             src_id = data.get('src_uid')  # Get the src_uid field
#             if src_id:  # If the src_uid exists in the data
#                 unique_src_ids.add(src_id)
#     return len(unique_src_ids)

# # Replace 'your_file.jsonl' with the path to your JSONL file
# jsonl_file = "benchmarks/merged_output.jsonl"
# print(f"Number of unique src_ids: {count_unique_src_ids(jsonl_file)}")


import json
from collections import defaultdict

def count_src_uid_samples(jsonl_file):
    src_uid_counts = defaultdict(int)  # Default dictionary to count occurrences of each src_uid
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            src_id = data.get('src_uid')  # Get the src_uid field
            if src_id:  # If src_uid exists in the data
                src_uid_counts[src_id] += 1
    return src_uid_counts

# Replace 'your_file.jsonl' with the path to your JSONL file
jsonl_file = "benchmarks/merged_output.jsonl"

src_uid_counts = count_src_uid_samples(jsonl_file)

# Print the number of samples for each unique src_uid
for src_id, count in src_uid_counts.items():
    print(f"src_uid: {src_id}, count: {count}")
