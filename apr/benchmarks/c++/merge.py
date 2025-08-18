import json

# 1. Read the C++ validation file
cpp_records = []
with open("C++.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        cpp_records.append(json.loads(line))

# 2. Read the problem_descriptions file
problems = {}
with open("problem_descriptions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        problems[item["src_uid"]] = item

# 3. Read the unittest_db file
with open("unittest_db.json", "r", encoding="utf-8") as f:
    tests = json.load(f)  # This file is a dict mapping src_uid → unit tests

# 4. Merge data based on src_uid
merged = []
for rec in cpp_records:
    src_uid = rec["src_uid"]
    problem_fields = problems.get(src_uid, {})
    merged_fields = {}
    for k, v in problem_fields.items():
        if k != "src_uid":
            merged_fields[f"prob_desc_{k}"] = v  # Add prefix "prob_desc_"
    # Convert sample inputs and outputs to JSON strings
    if "sample_inputs" in problem_fields:
        merged_fields["prob_desc_sample_inputs"] = json.dumps(problem_fields["sample_inputs"])
    if "sample_outputs" in problem_fields:
        merged_fields["prob_desc_sample_outputs"] = json.dumps(problem_fields["sample_outputs"])

    # Add hidden unit tests as JSON string
    if src_uid in tests:
        merged_fields["hidden_unit_tests"] = json.dumps(tests[src_uid])

    # Merge original C++ record with merged fields
    merged.append({
        **rec,            # All fields from the original C++ validation record
        **merged_fields   # Fields from problem_descriptions with prefix
    })

# 5. Save the merged output
with open("merged_cpp_validation.jsonl", "w", encoding="utf-8") as f:
    for item in merged:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Merged {len(merged)} records successfully.")
