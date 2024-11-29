import json
def combine_jsonl_files(implementations_file, test_cases_file, output_file):
    # Initialize the problems dictionary
    problems = {}
    
    # Read implementations file with utf-8 encoding
    with open(implementations_file, 'r', encoding='utf-8') as f:
        for line in f:
            impl = json.loads(line.strip())
            problem_id = impl['problem_id']
            
            # Initialize problem entry if it doesn't exist
            if problem_id not in problems:
                problems[problem_id] = {
                    "implementations": [],
                    "test_cases": []
                }
            
            # Add implementation
            problems[problem_id]["implementations"].append({
                "id": impl["id"],
                "buggy_submission_id": impl["buggy_submission_id"],
                "fixed_submission_id": impl["fixed_submission_id"],
                "user_id": impl["user_id"],
                "buggy_code": impl["buggy_code"],
                "fixed_code": impl["fixed_code"],
                "labels": impl.get("labels", []),  # Using get() in case labels is missing
                "change_count": impl["change_count"],
                "line_hunks": impl["line_hunks"]
            })
    
    # Read test cases file with utf-8 encoding
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        for line in f:
            test = json.loads(line.strip())
            problem_id = test["problem_id"]
            
            # Initialize problem entry if it doesn't exist
            if problem_id not in problems:
                problems[problem_id] = {
                    "implementations": [],
                    "test_cases": []
                }
            
            # Add test case, preserving id field
            problems[problem_id]["test_cases"].append({
                "id": test["id"],
                "input": test["input"],
                "output": test["output"]
            })
    
    # Write the combined data to output file with utf-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"problems": problems}, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    combine_jsonl_files(
        "ruby_train0.jsonl",
        "tests_all.jsonl",
        "combined_problems.json"
    )