import json
import random
import re

import json
import re
from typing import Union, List, Optional, Callable, Tuple
import json
import random
import re

def sample_n_random(items: List[str], n: int) -> List[str]:
    if n >= len(items):
        return items
    return random.sample(items, n)

def extract_json_fuzzy(output: str):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # پیدا کردن شروع لیست [
    match = re.search(r"\[\s*(\{.*)", output, re.DOTALL)
    if not match:
        raise ValueError("No JSON-like list found in the output.")

    content = match.group(1)

    # پیدا کردن تمام { ... }‌های بسته‌شده
    object_matches = list(re.finditer(r"\{.*?\}", content, re.DOTALL))

    if not object_matches:
        raise ValueError("No complete JSON objects found.")

    # ساختن لیست فقط از آبجکت‌های کامل
    valid_objects = []

    for obj_str in (m.group(0) for m in object_matches):
        def escape_json_string_fields(s):
            # This function finds string values and escapes newlines inside them
            def replacer(match):
                original_str = match.group(1)
                escaped_str = original_str.replace('\r', '\\r').replace('\n', '\\n')
                return f'"{escaped_str}"'
            
            return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', replacer, s)

        # Fix obj_str by escaping newlines inside strings
        fixed_obj_str_raw = escape_json_string_fields(obj_str)
        print(f"Fixed raw obj string:\n{fixed_obj_str_raw}\n")  # Debug

        try:
            # Now parse the fixed JSON string safely
            obj = json.loads(fixed_obj_str_raw)
            print(f"Parsed object: {obj}")  # Debug
            
            # Convert back to JSON string normalized
            fixed_obj_str = json.dumps(obj, ensure_ascii=False)
            valid_objects.append(fixed_obj_str)
            print(f"Fixed JSON string for final join:\n{fixed_obj_str}\n")  # Debug

        except json.JSONDecodeError:
            print(f"Skipping invalid JSON object after fixing: {obj_str}")  # Debug
            continue

    
    print("valid_objects:", valid_objects)
    # ساختن متن نهایی JSON
    json_string = "[\n" + ",\n".join(valid_objects) + "\n]"
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print("JSON string (invalid):")
        print(json_string)
        print("Decode error:", e)
        raise ValueError("Failed to recover a valid JSON list after trimming.") from e

outputs =  ['```json\n[\n    {\n        "input": "1 1\\n",\n        "output": "0"\n    },\n    {\n        "input": "2 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "3 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "4 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "5 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "6 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "7 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "8 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "9 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "10 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "11 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "12 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "13 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "14 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "15 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "16 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "17 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "18 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "19 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "20 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "21 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "22 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "23 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "24 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "25 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "26 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "27 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "28 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "29 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "30 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "31 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "32 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "33 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "34 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "35 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "36 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "37 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "38 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "39 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "40 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "41 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "42 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "43 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "44 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "45 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "46 1\\n",\n        "output": "1"\n    },\n    {\n        "input": "47 1\\n",\n        "output":']

all_tests = []

for output in outputs:
    try:
        print(f"Raw output: {output} (type: {type(output)})")
        unit_tests = extract_json_fuzzy(output)
        print("hiiiiii")
        print(f"unit_tests1: {unit_tests}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Test generation failed: {repr(e)}")
        all_tests.append([])
        continue

    if isinstance(unit_tests, list):
        print(f"unit_tests2: {unit_tests}")
        extracted = sample_n_random([
            (t["input"], t["output"])
            for t in unit_tests if isinstance(t, dict) and "input" in t and "output" in t
        ], max_num_tests)
        all_tests.append(extracted)
    else:
        all_tests.append([])

print(f"all_tests: {all_tests}")
