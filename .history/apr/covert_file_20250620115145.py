import json

def filter_ruby_entries(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                entry = json.loads(line)
                if entry.get("lang_cluster") == "Ruby":
                    outfile.write(json.dumps(entry) + '\n')
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")

# Example usage
filter_ruby_entries('output.jsonl', 'ruby_only.jsonl')
