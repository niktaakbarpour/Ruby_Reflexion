import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Adjust if needed
sys.path.append(project_root)
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
from middleware import retrieval
from middleware import prompt
from middleware.deepseek_local import Message
import re
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


# add near the top of the file if not already present
import uuid
import time

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

    # honor nsample
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


def gen(prompt_text, temperature, nsample, llm, model_name="deepseek"):
    cnt = 0
    while cnt < 999:
        try:
            # Select system prompt based on model_name
            system_prompt = (
                prompt.PROMPTS['system_decide']
                if 'claude' not in model_name
                else prompt.PROMPTS['system_decide_cs']
            )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt_text),
            ]

            raw = llm.generate_chat(messages, temperature=temperature, num_comps=nsample)
            print("get deepseek response......")

            # normalize + add OpenAI-style headers
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




# def gen(prompt_text, temperature, nsample, llm, model_name="deepseek"):
#     cnt = 0
#     while cnt < 999:
#         try:
#             # Select system prompt based on model_name
#             system_prompt = (
#                 prompt.PROMPTS['system_decide']
#                 if 'claude' not in model_name
#                 else prompt.PROMPTS['system_decide_cs']
#             )

#             messages = [
#                 Message(role="system", content=system_prompt),
#                 Message(role="user", content=prompt_text),
#             ]

#             # Generate using local DeepSeek model
#             c = llm.generate_chat(messages, temperature=temperature, num_comps=nsample)
#             print("get deepseek response......")
#             break
#         except Exception as e:
#             cnt += 1
#             time.sleep(5)
#             print(f"Gen Error: {e}")
#     else:
#         return None

#     # Add prompt field to mimic OpenAI structure
#     c["prompt"] = prompt_text
#     return c


# def gen(prompt_text, temperature, nsample, llm):
#     cnt = 0
#     while cnt < 999:
#         try:
#             messages = [
#                 Message(role="system", content=prompt.PROMPTS["system"]),
#                 Message(role="user", content=prompt_text),
#             ]
#             input_tokens = llm.prepare_prompt(messages)
#             full_prompt = input_tokens + "Assistant:"
#             tokens = llm.tokenizer.encode(full_prompt, return_tensors="pt").to(llm.model.device)
#             output_tokens = llm.model.generate(
#                 tokens,
#                 max_new_tokens=512,
#                 temperature=temperature,
#                 top_p=1.0,
#             )
#             break
#         except Exception as e:
#             cnt += 1
#             time.sleep(5)
#             print(f"Gen Error: {e}")
#     else:
#         return None

#     raw_text = llm.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

#     # 1) Try strict(ish) JSON from raw
#     content = try_extract_json(raw_text)
#     if content is None:
#         # 2) Heuristic fields from raw
#         fields = extract_decision_fields(raw_text)
#         if fields is None:
#             # 3) Last resort: run your code-cleaner & retry both
#             cleaned = llm.extract_output(raw_text)
#             fields = extract_decision_fields(cleaned)
#             if fields is None:
#                 content = json.dumps({
#                     "Target Language": "Unknown",
#                     "Justification": cleaned or raw_text
#                 })
#             else:
#                 content = json.dumps(fields)
#         else:
#             content = json.dumps(fields)

#     return {
#         "data": [{"content": content, "type": "text"}],
#         "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
#         "prompt": prompt_text
#     }



# PLACEHOLDER = re.compile(r'your (recommended language|reasoning)', re.I)

# def extract_decision_fields(text: str):
#     """
#     Pull TL/Justification ONLY from the Assistant block.
#     Returns dict or None.
#     """
#     # grab the Assistant block that contains "- Target Language:" lines
#     block_m = re.search(r'Assistant:\s*(.*?)(?:\n\s*User:|$)', text, re.S | re.I)
#     if not block_m:
#         return None
#     block = block_m.group(1)

#     # exact lines inside that block
#     tl_m = re.search(r'-\s*Target Language\s*:\s*([^\n]+)', block, re.I)
#     ju_m = re.search(r'-\s*Justification\s*:\s*(.+)', block, re.I | re.S)
#     if not tl_m or not ju_m:
#         return None

#     lang = tl_m.group(1).strip()
#     just = ju_m.group(1).strip()

#     # ignore placeholders
#     if PLACEHOLDER.search(lang) or PLACEHOLDER.search(just):
#         return None

#     return {
#         "Target Language": f"{lang}",
#         "Justification": f"{just}\n"
#     }


# def try_extract_json(text: str):
#     """
#     Try any real JSON blobs first; skip placeholders.
#     Fallback to regex extractor.
#     """
#     for m in re.finditer(r'\{[^{}]*\}', text, re.S):
#         cand = m.group(0)
#         try:
#             obj = json.loads(cand)
#         except Exception:
#             continue

#         if {"Target Language", "Justification"} <= obj.keys():
#             tl, ju = obj["Target Language"], obj["Justification"]
#             # skip placeholder JSON
#             if PLACEHOLDER.search(str(tl)) or PLACEHOLDER.search(str(ju)):
#                 continue
#             return json.dumps({
#                 "Target Language": f"[{tl}]",
#                 "Justification": f"[{ju}]\n"
#             })

#     # fallback
#     fields = extract_decision_fields(text)
#     return json.dumps(fields) if fields else None


def gen_request(prompt_text, temperature, nsample, llm):
    cnt = 0
    while True:
        if cnt == 999:
            return None
        try:
            messages = [
                {"role": "system", "content": prompt.PROMPTS["system"]},
                {"role": "user",   "content": prompt_text},
            ]
            prompt_tokens = llm.prepare_prompt(messages)
            output_tokens = llm.model.generate(
                prompt_tokens,
                max_new_tokens=4096,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
            )
            print("get deepseek response......")
            break
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"{e}")

    output_text = llm.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    message = llm.extract_output(output_text)

    return {
        "data": [
            {"content": message, "type": "text"}
        ],
        "prompt": prompt_text
    }


def _ensure_parsed_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v

def process_prompt(dt, bug_retrieval, temperature, mode, dec_dir, index, llm, dry_run=0):

    file_path = os.path.join(dec_dir, f"{index}_{temperature}_{dt['lang_cluster']}.json")
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
                
        lm_io = prompt.nohist(bug_retrieval) if mode == 'nohist' else prompt.decision(bug_retrieval)

        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            with open(file_path, "w") as f:
                f.write(json.dumps(lm_io[0], indent=4))
        else:
            out = gen(lm_io[0], temperature, 1, llm)
            export_data = {"oai_response": out, "source_data": dt}
            with open(file_path, "w") as f:
                f.write(json.dumps(export_data, indent=4))


def run(base_dir, num_proc, dry_run, it, mode, hist_top_k, dataset_path, llm):
    iter_dir = os.path.join(base_dir, f"iter_{it}")
    dec_dir = os.path.join(iter_dir, f"decide")
    if not os.path.exists(dec_dir):
        os.makedirs(dec_dir, exist_ok=True)

    apr_dataset = datasets.load_from_disk(dataset_path)
    # first_five = apr_dataset.select(range(5))
    # random_indices = random.sample(range(len(apr_dataset)), 5)
    # first_five = apr_dataset.select(random_indices)
    langs = ["Ruby"]
    # apr_dataset = apr_dataset.filter(lambda example: example["lang_cluster"] in langs)
    # apr_dataset = apr_dataset.select(range(2))

    # first_entry = apr_dataset.select(range(10))

    unfixed_file = os.path.join(iter_dir, "unfixed.json")
    if os.path.exists(unfixed_file):
        with open(unfixed_file, 'r') as f:
            data = json.load(f)
            unfixed_ids = list(data.keys())
    else:
        raise Exception("The intermediate file unfixed.json does not exist!")

    unfixed_dataset = apr_dataset.filter(lambda x: x['bug_code_uid'] in unfixed_ids)
    # temperature_list = np.linspace(0, 2, args.nsample)
    temperature_list = [0.3157894736842105]

    retrieval.init_vec_db(base_dir, dataset_path)
    retrieval.init_cos_similarity(base_dir)
    bug_properties, cos = retrieval.prepare_db(base_dir, apr_dataset)
    retrieval.update_pass_10(base_dir, it)
    
    decision_path = os.path.join(base_dir, f"iter_{it}/decision.json")

    decision_exist = os.path.exists(decision_path)
    if decision_exist:
        with open(decision_path, "r") as decision_file:
            decision_data = json.load(decision_file)
    for idx, dt in tqdm.tqdm(
    enumerate(unfixed_dataset),
    total=len(unfixed_dataset),
    desc=f"Preparing samples lang",
    ):
        for temperature in temperature_list:
            if decision_exist:
                uid = dt["bug_code_uid"]
                if uid in decision_data:
                    continue
            if mode == 'nohist':
                bug_retrieval = retrieval.retrieve(
                    base_dir, it, dt["bug_code_uid"], hist_top_k, bug_properties, cos, nohist=True
                )
            else:
                bug_retrieval = retrieval.retrieve(
                    base_dir, it, dt["bug_code_uid"], hist_top_k, bug_properties, cos
                )


            try:
                process_prompt(
                    dt,
                    bug_retrieval,
                    temperature,
                    mode,
                    dec_dir,
                    idx,
                    llm,
                    dry_run,
                )
            except Exception as e:
                print(f"Error occurred: {e}")

    # with concurrent.futures.ThreadPoolExecutor(
    #     max_workers=int(num_proc)
    # ) as executor:
    #     futures = []
    #     for idx, dt in tqdm.tqdm(
    #         enumerate(unfixed_dataset),
    #         total=len(unfixed_dataset),
    #         desc=f"Preparing samples lang",
    #     ):
    #         for temperature in temperature_list:
    #             if decision_exist:
    #                 uid = dt["bug_code_uid"]
    #                 if uid in decision_data:
    #                     continue
    #             if mode == 'nohist':
    #                 bug_retrieval = retrieval.retrieve(base_dir, it, dt["bug_code_uid"], hist_top_k, bug_properties, cos, nohist=True)
    #             else:
    #                 bug_retrieval = retrieval.retrieve(base_dir, it, dt["bug_code_uid"], hist_top_k, bug_properties, cos)
    #             future = executor.submit(
    #                 process_prompt,
    #                 dt,
    #                 bug_retrieval,
    #                 temperature,
    #                 mode,
    #                 dec_dir,
    #                 idx,
    #                 llm,
    #                 dry_run,
    #             )
    #             futures.append(future)

    #     for future in tqdm.tqdm(
    #         concurrent.futures.as_completed(futures),
    #         total=len(futures),
    #         desc=f"Calling OpenAI API",
    #     ):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Error occurred: {e}")


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
        "--hist_top_k",
        default=15,
        type=int,
        help="Top K historical data.",
    )
    parser.add_argument(
        "--dataset_path",
        default="/root/my/data/xCodeEval/apr",
        help="APR dataset path.",
    )
    args = parser.parse_args()
    # run(args.base_dir, args.num_proc, args.dry_run, args.it, args.hist_top_k, args.dataset_path)
    run(args.base_dir, args.num_proc, args.dry_run, args.it, args.mode, args.hist_top_k, args.dataset_path)

