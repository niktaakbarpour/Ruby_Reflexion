import json
import re
import ast
import time
import difflib
import copy
from promptsource.templates import Template


APR_PROMPT = """
Fix a buggy program written in {{lang_cluster}} language to solve the following programming problem:
Description: {{prob_desc_description}}
Input Specification: {{prob_desc_input_spec}}
Output Specification: {{prob_desc_output_spec}}
{% for input, output in zip(prob_desc_sample_inputs, prob_desc_sample_outputs) %}
Sample Input:
{{input}}
Sample Output:
{{output}}
{% endfor %}
Notes: {{prob_desc_notes}}
Take input from {{prob_desc_input_from}} and output to {{prob_desc_output_to}}

Here is the code with a bug of {{bug_exec_outcome}}:

{{bug_source_code}}

Provide the fixed {{lang_cluster}} code without any description or extra tokens.

Fixed source code:

||END-of-SRC|| 
"""

def code_truncate_regex(code):
    code_regex = r"```(.*?|)\n(?P<code>.*?)```"
    match = re.search(code_regex, code, re.DOTALL)
    code = match.group("code") if match else ""
    return code
    
def code_truncate(response):
    code = code_truncate_regex(response)
    if code == "":
        generation = response[response.find("def"):]
        tem = [s for s in generation.split('\n\n') if 'def ' in s or s[:1] == ' ']
        code = '\n\n'.join(tem).strip('```').strip()
    return code

def sanitize_code(code):
    prefixes = ["csharp", "cpp", "go", "javascript", "kotlin", "php", "python", "ruby", "rust", "c", "java", "json"]
    FLAG = True
    while FLAG == True:
        FLAG = False
        if code.startswith("```"):
            FLAG = True
            code = code.replace("```", "", 1)
        last_index = code.rfind("```")
        if last_index != -1:
            FLAG = True
            code = code[:last_index] + "" + code[last_index + len("```") :]
        for prefix in prefixes:
            if code.startswith(prefix):
                FLAG = True
                code = code.replace(prefix, "", 1)
                break
    return code

def extract_test_report(tests):
    # extract dict from the tests string
    test_report = sanitize_code(tests)
    return test_report

import json

# --- unchanged config ---
REQUIRED_FIELDS = [
    "bug_source_code",
    "prob_desc_description",
    "prob_desc_input_spec",
    "prob_desc_output_spec",
    "hidden_unit_tests",
    "lang",
]

OPTIONAL_OK_EMPTY = {
    "fix_code_uid", "fix_source_code", "fix_exec_outcome",
    "apr_id", "file_name", "tags", "prob_desc_notes",
    "prob_desc_sample_inputs", "prob_desc_sample_outputs",
    "prob_desc_time_limit", "prob_desc_memory_limit",
    "delete_cnt", "replace_cnt", "fix_ops_cnt", "equal_cnt",
    "insert_cnt", "difficulty", "similarity_score",
    "bug_exec_outcome", "lang_cluster", "potential_dominant_fix_op",
    "src_uid", "bug_code_uid", "prob_desc_created_at", "prob_desc_output_to",
    "prob_desc_input_from",
}

OPTIONAL_NUMERIC_ZERO_OK_TO_DROP = {
    "delete_cnt", "insert_cnt", "replace_cnt", "fix_ops_cnt",
}

# --- helpers ---
def _is_nan(x):
    try:
        return isinstance(x, float) and x != x
    except Exception:
        return False

def _is_empty(v, *, treat_zero_as_empty=False):
    if v is None:
        return True
    if _is_nan(v):
        return True
    if isinstance(v, (str, bytes)):
        s = v.strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    if treat_zero_as_empty and isinstance(v, (int, float)) and v == 0:
        return True
    return False

def _scrub_nans(x):
    if isinstance(x, dict):
        return {k: _scrub_nans(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_scrub_nans(v) for v in x]
    if _is_nan(x):
        return None
    return x

def _normalize_hidden_tests(hut_raw):
    import json
    hut = json.loads(hut_raw) if isinstance(hut_raw, str) else hut_raw
    if not isinstance(hut, list):
        raise ValueError("hidden_unit_tests must be a list")
    cleaned = []
    for i, t in enumerate(hut):
        if not isinstance(t, dict):
            raise ValueError(f"hidden_unit_tests[{i}] not a dict")
        inp = t.get("input", None)
        out = t.get("output", None)
        if out is not None and not isinstance(out, list):
            out = [out]
        def _valid_atom(x):
            if x is None or _is_nan(x): return False
            if isinstance(x, str) and x.strip() == "": return False
            return True
        if out is not None:
            out = [x for x in out if _valid_atom(x)]
        if _is_empty(inp) or _is_empty(out):
            raise ValueError(f"hidden_unit_tests[{i}] has empty input/output")
        cleaned.append({"input": inp, "output": out})
    return cleaned

# --- main sanitizer that *uses* REQUIRED/OPTIONAL and pads optionals ---
def sanitize_task_for_prompt(task: dict) -> dict:
    # 1) deep-scrub NaNs
    task = _scrub_nans(task)
    t = dict(task)  # shallow copy

    # 2) normalize HUT early so it counts for required check
    if "hidden_unit_tests" not in t:
        raise ValueError("hidden_unit_tests is required")
    t["hidden_unit_tests"] = _normalize_hidden_tests(t["hidden_unit_tests"])

    # 3) enforce REQUIRED_FIELDS: must exist and be non-empty
    missing = [k for k in REQUIRED_FIELDS if k not in t or _is_empty(t[k])]
    if missing:
        raise ValueError(f"Required fields missing/empty for APR prompt: {missing}")

    # 4) drop optional numeric zeros if desired
    for k in list(t.keys()):
        if k in OPTIONAL_NUMERIC_ZERO_OK_TO_DROP and isinstance(t[k], (int, float)) and t[k] == 0:
            del t[k]

    # 5) PAD *all* OPTIONAL_OK_EMPTY with a safe non-empty placeholder
    #    because the template may still call has_none_or_empty_raise() on them.
    PLACEHOLDER = "<none>"  # harmless text; won’t trip “empty”
    for k in OPTIONAL_OK_EMPTY:
        if k not in t or _is_empty(t[k], treat_zero_as_empty=True):
            # Use strings so Jinja checks pass; avoid 0/[]/{} which some templates treat as empty
            t[k] = PLACEHOLDER

    # 6) Finally, drop any other non-optional empties
    cleaned = {}
    for k, v in t.items():
        if k in OPTIONAL_OK_EMPTY:
            cleaned[k] = v
        else:
            if not _is_empty(v):
                cleaned[k] = v

    return cleaned

def find_empties_like_template(d):
    """Report empties for non-optional keys only (what would upset strict templates)."""
    empties = []
    def rec(path, v):
        top = path.split('.', 1)[0] if path else ""
        if top in OPTIONAL_OK_EMPTY:
            return
        if _is_empty(v, treat_zero_as_empty=True):
            empties.append(path or "<root>")
            return
        if isinstance(v, dict):
            for k, vv in v.items():
                rec(f"{path}.{k}" if path else k, vv)
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                rec(f"{path}[{i}]", vv)
    rec("", d)
    return empties

def prompt_xcodeeval(task):
    tpl = Template("apr", APR_PROMPT, "xCodeEval", delimeter="||END-of-SRC||")
    safe_task = sanitize_task_for_prompt(task)
    bad = find_empties_like_template(safe_task)
    
    for k in REQUIRED_FIELDS:
        v = safe_task.get(k, None)
        print(f"[REQ] {k}: type={type(v).__name__}, preview={repr(str(v)[:120])}")
    if safe_task["hidden_unit_tests"]:
        print("[REQ] HUT[0]:", safe_task["hidden_unit_tests"][0])

    return tpl.apply(safe_task)


def prompt_split_humaneval(prompt, mehotd_name):
    prompt = prompt.strip()
    prompt = prompt.replace("\r\n", "\n")
    before_func = prompt[:prompt.rfind("def ")]
    code = prompt[prompt.rfind("def "):]

    comment_start_1 = re.search("\"\"\"", code)
    comment_start_2 = re.search("\'\'\'", code)
    if comment_start_1:
        comment_start = comment_start_1.end()
    elif comment_start_2:
        comment_start = comment_start_2.end()


    example_start_1 = re.search("[eE]xample(:)?", code)
    example_start_2 = re.search("[fF]or [eE]xamble(:)?", code)
    example_start_3 = re.search(">>>", code)
    example_start_4 = re.search(mehotd_name+"\(.+\)", code[comment_start:])


    if example_start_1:
        comment = code[comment_start:example_start_1.start()]
        example = code[example_start_1.start():-4]
    elif example_start_2:
        comment = code[comment_start:example_start_2.start()]
        example = code[example_start_2.start():-4]
    elif example_start_3:
        comment = code[comment_start:example_start_3.start()]
        example = "Example:\n"+code[example_start_3.start():-4]
    elif example_start_4:
        comment = code[comment_start:example_start_4.start()+comment_start]
        example = "Example:\n"+code[example_start_4.start()+comment_start:-4]
    else:
        comment = code[comment_start:-4]
        example = ""
    comment = comment.strip().replace("\n", " ")
    comment = re.sub("\s+", " ", comment)

    example = re.sub("\n(\s)*","\n\t",example)
    test_case = "\t"+example.strip()
    signature = code[:code.index("\n")+1]

    return before_func, signature, comment, test_case

def build_test_method(test_list, test_imports, method_name):
    if test_imports:
        test_imports = "\n".join(test_imports)
        test_method = test_imports + "\n"
    else:
        test_method = ""
    test_method = "def check(" + method_name + "):\n"
    if len(test_list) == 0:
        return test_method + "\treturn True" + "\n"
    for test in test_list:
        test_method += '\t' + test + "\n"
    return test_method.strip("\n")

def find_method_name(code, lang="python"):
    try:
        parsed = ast.parse(code)
        function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        if function_defs:
            if len(function_defs) == 1:
                method_name = function_defs[0].name
            else:
                method_name = function_defs[-1].name if function_defs[-1].name != "main" else function_defs[-2].name
        else:
            method_name = None
    except:
        method_name = None

    return method_name


def code_split(func):
    '''
    Split code into signature, comment and function body
    '''
    func = func.replace("\r\n", "\n")
    before_func = func[:func.rfind("def ")]
    code = func[func.rfind("def "):]

    is_comment = False
    comments = []
    
    statements = code.split("\n")
    for s_idx, s in enumerate(statements):
        s = s.strip()
        if s.startswith("def"):
            signature = statements[:s_idx+1]
            method_name = s.split("def ")[1].split("(")[0]
            func_body_idx = s_idx+1
            tmp_statement = statements[func_body_idx].strip()
            if not tmp_statement.startswith("'''"):
                break
        elif s.startswith("'''") and not is_comment:
            is_comment = True

        elif is_comment:
            if s.startswith("'''"):
                is_comment = False
                func_body_idx = s_idx+1
                break
            comments.append(s)
    func_body = statements[func_body_idx:]
    return method_name, "\n".join(signature), "\n".join(comments), "\n".join(func_body), before_func

def construct_system_message(requirement, role, team=''):
    if team == '':
        system_message = "The requirement from users is: \n{'requirement':\n"  +  "'"+ requirement.replace('\n\n','\n').strip(".") + "'\n}\n\n" + role
    else:
        system_message = team + '\n '+ \
                    "The requirement from users is: \n{'requirement':\n"  +  "'"+ requirement.replace('\n\n','\n').strip(".") + "'\n}\n\n" + \
                    role
                
    return system_message
    