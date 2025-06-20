import json

def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_json(file_path):
    """Load a JSON file into a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_files(ruby_file, problems_file, unittest_file, output_file):
    """Merge JSONL and JSON files based on src_uid and save the result to a new JSONL file."""
    ruby_data = load_jsonl(ruby_file)
    problems_data = {entry["src_uid"]: entry for entry in load_jsonl(problems_file)}
    unittest_data = load_json(unittest_file)
    
    merged_data = []
    for ruby_entry in ruby_data:
        src_uid = ruby_entry.get("src_uid")
        
        if src_uid in problems_data:
            merged_entry = {**ruby_entry, **problems_data[src_uid]}  # Merge Ruby and problem description
        else:
            merged_entry = ruby_entry  # No problem description found
        
        if src_uid in unittest_data:
            merged_entry["unittest_cases"] = unittest_data[src_uid]  # Add unittest cases
        
        merged_data.append(merged_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Merged data saved to {output_file}")

# Example usage
merge_files("ruby_validation.jsonl", "problem_descriptions.jsonl", "unittest_db.json", "validation_output.jsonl")
