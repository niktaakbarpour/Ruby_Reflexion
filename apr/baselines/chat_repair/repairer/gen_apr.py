import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Adjust if needed
sys.path.append(project_root)
import time
import tqdm
import json
# import openai
# from openai import OpenAI
import argparse
import datasets
import concurrent
import numpy as np
from promptsource.templates import Template
from middleware.deepseek_local import Message
# import requests
from middleware import prompt
from datasets import Dataset
import re
import random
# Load model directly

SHORT_LANG_MAP = {
    "GNU C++": "C++",
    "GNU C++17": "C++",
    "MS C++ 2017": "C++",
    "MS C++": "C++",
    "Java 8": "Java",
    "Java 6": "Java",
    "GNU C++11": "C++",
    "Java 11": "Java",
    "GNU C++14": "C++",
    "Mono C#": "C#",
    "GNU C": "C",
    "Python 3": "Python",
    "PyPy 3": "Python",
    "GNU C11": "C",
    "Go": "Go",
    "Rust": "Rust",
    "PyPy 2": "Python",
    "Python 2": "Python",
    "MS C#": "C#",
    "Kotlin": "Kotlin",
    "GNU C++0x": "C++",
    "Java 7": "Java",
    "Node.js": "Javascript",
    ".NET Core C#": "C#",
    "PHP": "PHP",
    "GNU C++17 Diagnostics": "C++",
    "Clang++17 Diagnostics": "C++",
    "JavaScript": "Javascript",
    "Ruby": "Ruby",
    "C# 10": "C#",
    "C# 8": "C#",
    "Clang++20 Diagnostics": "C++",
    "GNU C++17 (64)": "C++",
    "GNU C++20 (64)": "C++",
    "Java 17": "Java",
    "Kotlin 1.4": "Kotlin",
    "Kotlin 1.5": "Kotlin",
    "Kotlin 1.6": "Kotlin",
    "Kotlin 1.7": "Kotlin",
    "PyPy 3-64": "Python",
    "Python 3 + libs": "Python",
    "Ruby 3": "Ruby",
    "Rust 2021": "Rust",
}

LANGS = sorted(set([v for k, v in SHORT_LANG_MAP.items()]))


# openai.api_key = os.environ["API_KEY"]
# openai.api_base = os.environ["API_BASE"]
# model_name = os.environ["MODEL_NAME"]



import time
import uuid

def _normalize_to_openai_like(raw, nsample):
    """
    Turn whatever `llm.generate_chat(...)` returns into:
    {"choices": [{"index": i, "message": {"role":"assistant","content": ...}, "finish_reason":"stop"}, ...]}
    """
    # collect assistant contents
    contents = []

    if isinstance(raw, str):
        contents = [raw]
    elif isinstance(raw, tuple):
        # e.g., (full_prompt, content)
        if len(raw) == 2:
            _, content = raw
            contents = [content]
        else:
            contents = [str(raw)]
    elif isinstance(raw, (list, tuple)):
        # list of strings or objects with "content"
        for item in raw:
            if isinstance(item, str):
                contents.append(item)
            elif isinstance(item, dict):
                # Try common shapes
                if "message" in item and isinstance(item["message"], dict) and "content" in item["message"]:
                    contents.append(item["message"]["content"])
                elif "content" in item:
                    contents.append(item["content"])
                else:
                    contents.append(str(item))
            else:
                contents.append(str(item))
    elif isinstance(raw, dict):
        if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
            # Already OpenAI-like; ensure minimal fields
            choices = []
            for i, c in enumerate(raw["choices"]):
                msg = c.get("message", {"role": "assistant", "content": c.get("content", "")})
                if "role" not in msg:
                    msg["role"] = "assistant"
                if "content" not in msg:
                    msg["content"] = c.get("content", "")
                choices.append({
                    "index": c.get("index", i),
                    "message": msg,
                    "finish_reason": c.get("finish_reason", "stop"),
                })
            return {"choices": choices}
        elif "content" in raw:
            contents = [raw["content"]]
        else:
            contents = [str(raw)]
    else:
        contents = [str(raw)]

    # enforce nsample count
    if not isinstance(nsample, int) or nsample <= 0:
        nsample = 1
    if len(contents) > nsample:
        contents = contents[:nsample]
    while len(contents) < nsample:
        contents.append(contents[-1] if contents else "")

    choices = []
    for i, content in enumerate(contents):
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        })
    return {"choices": choices}


def gen(prompt_text, temperature, nsample, llm, model_name="deepseek-local"):
    cnt = 0
    while cnt < 999:
        try:
            messages = [
                Message(role="system", content=prompt.PROMPTS["system"]),
                Message(role="user", content=prompt_text),
            ]
            raw = llm.generate_chat(messages, temperature=temperature, num_comps=nsample)

            # normalize to choices
            norm = _normalize_to_openai_like(raw, nsample)

            # fabricate OpenAI headers
            c = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": norm["choices"],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                },
                "prompt": prompt_text,
            }

            # optional: extract code from first choice
            try:
                content = c["choices"][0]["message"]["content"]
                fixed = llm.extract_output(content)
                c["fixed_source_code"] = fixed
            except Exception as e:
                print(f"Extract error: {e}")
                c["fixed_source_code"] = ""

            # debug prints
            try:
                print("### MODEL CONTENT (first 200):", content[:200] if isinstance(content, str) else str(content)[:200])
                print("### FIXED CODE (first 200):", c["fixed_source_code"][:200])
            except Exception:
                pass

            return c
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"Gen Error: {e}")

    return None




# def gen(prompt_text, temperature, nsample, llm):
#     cnt = 0
#     while cnt < 999:
#         try:
#             messages = [
#                 Message(role="system", content=prompt.PROMPTS["system"]),
#                 Message(role="user", content=prompt_text),
#             ]
#             raw = llm.generate_chat(messages, temperature=temperature, num_comps=nsample)

#             # --- Normalize to OpenAI-like dict ---
#             if isinstance(raw, tuple) and len(raw) == 2:
#                 # e.g., (full_prompt, content)
#                 _, content = raw
#                 c = {"choices": [{"message": {"role": "assistant", "content": content}}]}
#             elif isinstance(raw, str):
#                 c = {"choices": [{"message": {"role": "assistant", "content": raw}}]}
#             elif isinstance(raw, dict) and "choices" in raw:
#                 c = raw
#             else:
#                 # last resort
#                 c = {"choices": [{"message": {"role": "assistant", "content": str(raw)}}]}

#             break
#         except Exception as e:
#             cnt += 1
#             time.sleep(5)
#             print(f"Gen Error: {e}")
#     else:
#         return None

#     # Optional: extract code
#     try:
#         content = c["choices"][0]["message"]["content"]
#         fixed = llm.extract_output(content)  # your extractor
#         c["fixed_source_code"] = fixed
#     except Exception as e:
#         print(f"Extract error: {e}")
#         c["fixed_source_code"] = ""

#     c["prompt"] = prompt_text

#     # Print the fields you care about (not the raw object)
#     print("### MODEL CONTENT (first 200):", content[:200] if isinstance(content, str) else str(content)[:200])
#     print("### FIXED CODE (first 200):", c["fixed_source_code"][:200])
#     return c


def gen_request(prompt_text, temperature, nsample, llm):
    """
    Alias of gen(), preserving interface.
    """
    return gen(prompt_text, temperature, nsample)

def _ensure_parsed_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v



def process_prompt(dt, temperature, nsample, output_dir, index, attempt, llm, dry_run=0):
    language = dt["lang_cluster"]
    file_path = os.path.join(output_dir, f"{index}_{attempt}_{temperature}_{language}.json")
    if not os.path.exists(file_path):
        dt["prob_desc_sample_inputs"]  = _ensure_parsed_list(dt.get("prob_desc_sample_inputs"))
        dt["prob_desc_sample_outputs"] = _ensure_parsed_list(dt.get("prob_desc_sample_outputs"))

        def _coerce_empty(d, key, default):
            v = d.get(key, None)
            if v is None:
                d[key] = default
            elif isinstance(v, str) and v.strip() == "":
                d[key] = default
            elif isinstance(v, (list, dict)) and len(v) == 0:
                d[key] = default

        # Fields that are allowed to be empty pre-repair but must not be ""/None for the template
        _coerce_empty(dt, "fix_source_code", "/* empty before repair */")
        _coerce_empty(dt, "fix_code_uid", "NA")
        _coerce_empty(dt, "fix_exec_outcome", "NA")
        _coerce_empty(dt, "prob_desc_notes", "NA")

        # If your template references any of these, keep them safe too
        for k in ("prob_desc_description", "prob_desc_input_spec", "prob_desc_output_spec"):
            _coerce_empty(dt, k, "N/A")

        def _empty_fields(d):
            bad = {}
            for k, v in d.items():
                if v is None:
                    bad[k] = "None"
                elif isinstance(v, str) and v.strip() == "":
                    bad[k] = "empty-str"
                elif isinstance(v, (list, dict)) and len(v) == 0:
                    bad[k] = "empty-" + ("list" if isinstance(v, list) else "dict")
            return bad

        # --- just before calling prompt.apr(dt) ---
        empties = _empty_fields(dt)
        if empties:
            ctx = {k: dt.get(k) for k in ("_id", "uid", "src_uid", "lang_cluster") if k in dt}
            print(f"[DEBUG] Empty fields for sample {ctx}:")
            for k, reason in sorted(empties.items()):
                print(f"   - {k}: {reason}")

        lm_io = prompt.apr(dt)
        assert isinstance(lm_io, str) or (isinstance(lm_io, list) and len(lm_io) >= 1)
        if dry_run:
            with open(file_path, 'w') as f:
                f.write(json.dumps(lm_io, indent=4))
        else:
            out = gen(lm_io, temperature, nsample, llm)
            export_data = {"oai_response": out, "source_data": dt}
            with open(file_path, 'w') as f:
                f.write(json.dumps(export_data, indent=4))

def run(base_dir, num_proc, dry_run, nsample, nattempt, temperature, dataset_path, llm):
    output_dir = os.path.join(base_dir, "repair")
    os.makedirs(output_dir, exist_ok=True)
    apr_dataset = datasets.load_from_disk(dataset_path)
    # first_five = apr_dataset.select(range(5))
    random_indices = random.sample(range(len(apr_dataset)), 10)
    first_five = apr_dataset.select(random_indices)

    print("Bypassing language filter for single-row test")
    temperature_list = [temperature]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=int(num_proc)) as executor:
    #     futures = []

    for idx, dt in tqdm.tqdm(
        enumerate(first_five.to_list()),
        total=len(first_five.to_list()),
        desc=f"Preparing samples lang",
    ):
        for attempt in range(nattempt):
            for temperature in temperature_list:
                x = process_prompt(dt, temperature, nsample,
                output_dir, idx, attempt, llm, dry_run)
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="dumped/oai/apr_n_sample_20",
        help="Path to the trans-repair base directory.",
    )
    parser.add_argument(
        "--num-proc",
        default=1,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--dry-run",
        default=0,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--nsample",
        default=1,
        type=int,
        help="Number of parallel API request.",
    )
    # deepseek only allows nsample=1 currently, use this as the number of repetitive generation for each problem
    parser.add_argument(
        "--nattempt",
        default=20,
        type=int,
        help="Number of attempts of generation for each problem.",
    )
    parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Set the temperature for the language model."
    )
    args = parser.parse_args()
    run(args.base_dir, args.num_proc, args.dry_run, args.nsample, args.nattempt, args.temperature)

