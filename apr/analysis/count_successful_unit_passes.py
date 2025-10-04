import json
# Counts how many records in the JSONL file achieved at least one pass@1_unit_iter[i] == 1.0 (for iterations 0–10) and reports both the raw count and its fraction of 343.
file_path = ""  # Replace with your actual file name
count = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        for i in range(11):  # from 0 to 10 inclusive
            key = f"pass@1_unit_iter{i}"
            if obj.get(key) == 1.0:
                count += 1
                break  # No need to check other keys for this object

print("Number of objects with pass@1_unit_iter{i} == 1.0 for any i from 0 to 10:", count)

print(count/343)