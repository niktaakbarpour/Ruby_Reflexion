import json
import jsonlines

# File paths
ruby_validation_path = "ruby_validation.jsonl"
problem_descriptions_path = "problem_descriptions.jsonl"
unittest_path = "unittest_db.json"
output_path = "validation_output.jsonl"

# Load ruby_validation.jsonl
ruby_validation_data = {}
with jsonlines.open(ruby_validation_path) as reader:
    for obj in reader:
        ruby_validation_data[obj["src_uid"]] = obj

# Load problem_descriptions.jsonl
problem_descriptions_data = {}
with jsonlines.open(problem_descriptions_path) as reader:
    for obj in reader:
        problem_descriptions_data[obj["src_uid"]] = obj

# Load unittest.json
with open(unittest_path, "r", encoding="utf-8") as f:
    content = f.read()

# Handle non-standard JSON syntax if necessary
# If the file uses a format like: unittest_db = { ... }
if content.strip().startswith("unittest_db"):
    content = content.split("=", 1)[1].strip()

unittest_data = json.loads(content)

# Merge based on src_uid
with jsonlines.open(output_path, "w") as writer:
    for src_uid in ruby_validation_data:
        merged_entry = {"src_uid": src_uid}

        # Add ruby validation fields
        merged_entry.update(ruby_validation_data.get(src_uid, {}))

        # Add problem description fields
        if src_uid in problem_descriptions_data:
            merged_entry["problem_description"] = problem_descriptions_data[src_uid]
        else:
            merged_entry["problem_description"] = None

        # Add unit tests
        merged_entry["unit_tests"] = unittest_data.get(src_uid, [])

        writer.write(merged_entry)

print(f"Merged data written to {output_path}")
