import os
import json
import uuid
import time

# This script scans JSON files in a directory and, if missing, inserts a fabricated OpenAI-style response ID and metadata before saving updated copies to a new folder.
def add_ids_to_json_files(input_dir, output_dir):
    # Create the new directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        with open(input_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Skipping {filename}, JSON error: {e}")
                continue

        if "oai_response" in data and "id" not in data["oai_response"]:
            # Fabricate an OpenAI-like ID
            fake_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            data["oai_response"]["id"] = fake_id
            data["oai_response"]["object"] = "chat.completion"
            data["oai_response"]["created"] = int(time.time())
            data["oai_response"]["model"] = "deepseek-local"

            # Optional: add a minimal usage block
            if "usage" not in data["oai_response"]:
                data["oai_response"]["usage"] = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                }

            print(f"Updated {filename} with id={fake_id}")
        else:
            print(f"Skipped {filename}, already has id")

        # Save result into the new directory
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# Example usage:
# Creates "updated_results" folder inside the same parent directory
input_dir = ""
output_dir = os.path.join(os.path.dirname(input_dir), "updated_results")
add_ids_to_json_files(input_dir, output_dir)
