import json
import os

def convert_jsonl_to_execeval_format(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    written = 0

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for idx, record in enumerate(records):
        total += 1
        bug_uid = record.get("bug_code_uid", f"unknown_{total}")
        lang = "Ruby"
        src_uid = record.get("src_uid", "")

        # --- Find passing implementation ---
        passing_idx = None
        for i in range(11):
            key = f"pass@1_unit_iter{i}"
            if record.get(key) == 1.0:
                passing_idx = i
                break

        implementations = record.get("implementations", [])
        if passing_idx is not None and passing_idx < len(implementations):
            raw_code = implementations[passing_idx]
            passed = True
        elif implementations:
            raw_code = implementations[-1]
            passed = False
        else:
            print(f"⚠️ Skipping {bug_uid}: no implementations")
            continue

        # ✅ Wrap in code block
        fixed_code = f"```ruby\n{raw_code}\n```"


        # --- Prepare unit test results ---
        unit_tests = record.get("hidden_unit_tests", [])
        unit_test_results = []
        if passed:
            unit_test_results = [[{"exec_outcome": "PASSED"} for _ in unit_tests]]
        else:
            unit_test_results = [[{"exec_outcome": "FAILED"} for _ in unit_tests]]

        # --- Build final record ---
        source_data = {
            "delete_cnt": record.get("delete_cnt", 0),
            "fix_code_uid": record.get("fix_code_uid", ""),
            "difficulty": record.get("difficulty", ""),
            "replace_cnt": record.get("replace_cnt", 0),
            "fix_ops_cnt": record.get("fix_ops_cnt", 0),
            "prob_desc_output_spec": record.get("prob_desc_output_spec", ""),
            "apr_id": record.get("apr_id", ""),
            "bug_source_code": record.get("bug_source_code", ""),
            "prob_desc_input_spec": record.get("prob_desc_input_spec", ""),
            "src_uid": src_uid,
            "bug_code_uid": bug_uid,
            "equal_cnt": record.get("equal_cnt", 0),
            "prob_desc_time_limit": record.get("prob_desc_time_limit", ""),
            "similarity_score": record.get("similarity_score", 0),
            "bug_exec_outcome": record.get("bug_exec_outcome", ""),
            "lang_cluster": record.get("lang_cluster", ""),
            "potential_dominant_fix_op": record.get("potential_dominant_fix_op", ""),
            "prob_desc_sample_inputs": record.get("prob_desc_sample_inputs", ""),
            "prob_desc_input_from": record.get("prob_desc_input_from", ""),
            "prob_desc_description": record.get("prob_desc_description", ""),
            "lang": lang,
            "fix_exec_outcome": record.get("fix_exec_outcome", ""),
            "insert_cnt": record.get("insert_cnt", 0),
            "fix_source_code": fixed_code,
            "prob_desc_notes": record.get("prob_desc_notes", ""),
            "file_name": record.get("file_name", ""),
            "tags": record.get("tags", []),
            "prob_desc_output_to": record.get("prob_desc_output_to", ""),
            "prob_desc_created_at": record.get("prob_desc_created_at", ""),
            "prob_desc_memory_limit": record.get("prob_desc_memory_limit", ""),
            "prob_desc_sample_outputs": record.get("prob_desc_sample_outputs", ""),
            "hidden_unit_tests": unit_tests,
        }

        output_record = {
            "oai_response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": fixed_code
                        }
                    }
                ],
                "prompt": "",
                "usage": {}
            },
            "source_data": source_data
        }

        # --- New file name format ---
        file_name = f"{idx}_{bug_uid}_{lang}.json"
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(json.dumps(output_record, ensure_ascii=False, indent=2) + "\n")
        written += 1

    print(f"✅ Total entries: {total}, Converted: {written}, Skipped: {total - written}")

# Example usage:
convert_jsonl_to_execeval_format("../results/ds.jsonl", "repair/")
