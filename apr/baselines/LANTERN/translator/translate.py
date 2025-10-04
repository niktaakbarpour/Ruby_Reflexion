import os
import time
import tqdm
import json
# import openai
# import requests
# from openai import OpenAI
import argparse
import datasets
import concurrent
import numpy as np
from promptsource.templates import Template
from analyzer.decision import TransDecision
from middleware.retrieval import build_target_db
from middleware.history import get_last_incorrect_samples, cp_last_incorrect_samples
from middleware import prompt
import shutil
from middleware.deepseek_local import Message
import random

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
    Coerce `raw` from llm.generate_chat(...) into:
    {"choices": [{"index": i, "message": {"role":"assistant","content": ...}, "finish_reason":"stop"}, ...]}
    """
    contents = []

    if isinstance(raw, str):
        contents = [raw]
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                contents.append(item)
            elif isinstance(item, dict):
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
            # Already OpenAI-like: ensure minimal fields and return
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

    # Honor nsample
    if not isinstance(nsample, int) or nsample <= 0:
        nsample = 1
    if len(contents) > nsample:
        contents = contents[:nsample]
    while len(contents) < nsample:
        contents.append(contents[-1] if contents else "")

    return {
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for i, content in enumerate(contents)
        ]
    }

def gen(prompt_text, temperature, nsample, llm, model_name="deepseek-local"):
    cnt = 0
    while cnt < 999:
        try:
            # Construct messages like OpenAI API (your local wrapper expects Message objects)
            messages = [Message(role="user", content=prompt_text)]

            # Generate with local model
            raw = llm.generate_chat(
                messages,
                temperature=temperature,
                num_comps=nsample
            )
            print("get deepseek response......")

            # Normalize + add OpenAI-style headers
            norm = _normalize_to_openai_like(raw, nsample)
            out = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": norm["choices"],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                "prompt": prompt_text,
            }
            return out
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"Gen Error: {e}")

    return None





# def gen(prompt_text, temperature, nsample, llm):
#     cnt = 0
#     while cnt < 999:
#         try:
#             # Construct the messages list like OpenAI's API
#             messages = [Message(role="user", content=prompt_text)]
            
#             # Call the local model to generate responses
#             c = llm.generate_chat(
#                 messages,
#                 temperature=temperature,
#                 num_comps=nsample
#             )
#             print("get deepseek response......")
#             break
#         except Exception as e:
#             cnt += 1
#             time.sleep(5)
#             print(f"Gen Error: {e}")
#     else:
#         return None

#     # Add prompt to match OpenAI's structure
#     c["prompt"] = prompt_text
#     return c


# Alias preserving interface
def gen_request(prompt_text, temperature, nsample, llm):
    res = gen(prompt_text, temperature, nsample, llm)
    if res is None:
        return None
    return {"data": [{"content": res['choices'][0]['message']['content'], "type": "text"}], "prompt": prompt_text}

def _ensure_parsed_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v

def process_prompt(dt, temperature, trans_dir, target_lang, index, r_mode, llm, dry_run=0):
    dt["source_lang"] = dt["lang_cluster"]
    dt["target_lang"] = target_lang
    language = f"{dt['source_lang']}--{dt['target_lang']}"
    file_path = os.path.join(trans_dir, f"{index}_{temperature}_{language}.json")
    if not os.path.exists(file_path):
        if r_mode != "ultimate2":
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
        lm_io = prompt.trans(dt)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
        else:
            out = gen(lm_io[0], temperature, 1, llm)
            # out = gen_request(lm_io[0], temperature, 1)
            export_data = {"oai_response": out, "source_data": dt}
            open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")



def run(base_dir, num_proc, dry_run, it, mode, r_mode, dataset_path, llm, config_path=""):
    
    iter_dir = os.path.join(base_dir, f"iter_{it}")
    unfixed_file = os.path.join(iter_dir, "unfixed.json")
    if os.path.exists(unfixed_file):
        with open(unfixed_file, 'r') as f:
            # unfixed_ids = list(json.load(f).keys())
            data = json.load(f)
            unfixed_ids = list(data.keys())
    else:
        raise FileNotFoundError("unfixed.json not found")

    trans_dir = os.path.join(iter_dir, f"trans")
    if not os.path.exists(trans_dir):
        os.makedirs(trans_dir, exist_ok=True)

    # copy mode
    if mode == "copy":
        print('copying...')
        cp_last_incorrect_samples(base_dir, it, unfixed_ids)
        return

    # decision-based translation
    decision = TransDecision(base_dir, it, config_path)
    if mode in ["reasoning", "nohist", "nocot"]:
        build_target_db(base_dir, it)

    apr_dataset = datasets.load_from_disk(dataset_path)

    # first_five = apr_dataset.select(range(5))
    # random_indices = random.sample(range(len(apr_dataset)), 5)
    # first_five = apr_dataset.select(random_indices)

    # first_entry = apr_dataset.select(range(10))

    if r_mode in["ultimate2", "testfailure"]:
        unfixed_dataset = get_last_incorrect_samples(base_dir, it, unfixed_ids)
    else:
        unfixed_dataset = apr_dataset.filter(lambda x: x['bug_code_uid'] in unfixed_ids)

    temperature_list = [0.3157894736842105]
    # with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_proc)) as executor:
    #     futures = []
    for idx, dt in enumerate(tqdm.tqdm(unfixed_dataset, desc="Preparing samples lang")):
        target_lang = decision.decide_lang(sample=dt, it=it, mode=mode)
        for temp in temperature_list:
            process_prompt(dt, temp, trans_dir, target_lang, idx, r_mode, llm, dry_run)
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
        "--it",
        default=1,
        type=int,
        help="Current iteration epoch of trans-repair.",
    )
    parser.add_argument(
        "--mode",
        default="vanilla",
        help="Translation mode.",
    )
    parser.add_argument(
        "--r_mode",
        default="vanilla",
        help="Repair mode.",
    )
    args = parser.parse_args()
    run(args.base_dir, args.num_proc, args.dry_run, args.it, args.mode, args.r_mode)

