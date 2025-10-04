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
from middleware.repair_retrieval import add_hist, construct_conversation
from middleware.history import load_last_repair, load_last_tests, get_last_incorrect_samples_cr
from middleware import prompt
from middleware.deepseek_local import DeepSeekCoder
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
    Convert various possible returns from llm.generate_chat(...) into:
    {
      "choices": [
        {"index": i, "message": {"role": "assistant", "content": "..."},
         "finish_reason": "stop"}
        ...
      ]
    }
    """
    # Convert raw into a list of assistant contents (length == nsample if possible)
    contents = []

    # Common cases: str, list[str], dict with "choices"
    if isinstance(raw, str):
        contents = [raw]
    elif isinstance(raw, (list, tuple)):
        # e.g., list of strings or list of dicts
        for item in raw:
            if isinstance(item, str):
                contents.append(item)
            elif isinstance(item, dict) and "message" in item and "content" in item["message"]:
                contents.append(item["message"]["content"])
            elif isinstance(item, dict) and "content" in item:
                contents.append(item["content"])
            else:
                contents.append(str(item))
    elif isinstance(raw, dict):
        if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
            # already OpenAI-like or near
            return {
                "choices": [
                    {
                        "index": c.get("index", i),
                        "message": c.get("message", {"role": "assistant", "content": c.get("content", "")}),
                        "finish_reason": c.get("finish_reason", "stop"),
                    }
                    for i, c in enumerate(raw["choices"])
                ]
            }
        elif "content" in raw:
            contents = [raw["content"]]
        else:
            contents = [str(raw)]
    else:
        contents = [str(raw)]

    # Ensure at least 1; trim or pad to nsample length
    if nsample is None or nsample <= 0:
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


def gen(prompt_text, temperature, nsample, mode, llm, msg=None, model_name="deepseek-local"):
    """
    Generate a response using local DeepSeekCoder, and return an object that
    matches OpenAI's ChatCompletion format as closely as possible.
    """
    cnt = 0

    # Prepare messages
    if mode == "ultimate2":
        messages = msg if msg is not None else []
    else:
        messages = [
            {"role": "system", "content": prompt.PROMPTS['system']},
            {"role": "user", "content": prompt_text},
        ]
        print(f"messages: {messages}")

    while cnt < 999:
        try:
            # Convert messages to the structure expected by DeepSeekCoder
            msg_objs = [Message(role=m["role"], content=m["content"]) for m in messages]
            print(f"msg_objs: {msg_objs}")

            # Generate responses
            raw = llm.generate_chat(
                msg_objs,
                temperature=temperature,
                num_comps=nsample
            )
            print(f"raw: {raw}")

            # Normalize to OpenAI-like "choices"
            norm = _normalize_to_openai_like(raw, nsample)
            print(f"norm: {norm}")

            # Fabricate OpenAI metadata
            out = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": norm["choices"],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                }
            }
            print(f"out: {out}")

            # Attach your prompt/conversation for traceability
            if mode == "ultimate2":
                out["conversation"] = messages
            else:
                out["prompt"] = prompt_text

            return out
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"Gen Error: {e}")

    return None

# def gen(prompt_text, temperature, nsample, mode, llm, msg=None):
#     """
#     Generate a response using local DeepSeekCoder, matching OpenAI's ChatCompletion format.
#     """
#     cnt = 0

#     # Prepare messages
#     if mode == "ultimate2":
#         messages = msg  # Already formatted messages
#     else:
#         messages = [
#             {"role": "system", "content": prompt.PROMPTS['system']},
#             {"role": "user", "content": prompt_text},
#         ]

#     while cnt < 999:
#         try:
#             # Convert messages to the structure expected by DeepSeekCoder
#             msg_objs = [Message(role=m["role"], content=m["content"]) for m in messages]

#             # Generate responses
#             c = llm.generate_chat(
#                 msg_objs,
#                 temperature=temperature,
#                 num_comps=nsample
#             )
#             break
#         except Exception as e:
#             cnt += 1
#             time.sleep(5)
#             print(f"Gen Error: {e}")
#     else:
#         return None

#     # Add metadata to mimic OpenAI response
#     if mode == "ultimate2":
#         c["conversation"] = msg
#     else:
#         c["prompt"] = prompt_text
#     return c


def gen_request(prompt_text, temperature, nsample, llm):
    """
    Alias of gen(), preserving interface.
    """
    return gen(prompt_text, temperature, nsample, llm)


def _ensure_parsed_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def process_prompt(dt, temperature, nsample, output_dir, index, attempt, mode, llm, msg=None, dry_run=0):
    language = dt["lang_cluster"]
    if mode in ["ultimate2", "testfailure"]:
        uid = dt["bug_code_uid"]
        file_path = os.path.join(output_dir, f"{index}_{uid}_{temperature}_{language}.json")
    else:
        file_path = os.path.join(output_dir, f"{index}_{attempt}_{temperature}_{language}.json")
        print(f"file_path: {file_path}")

    if mode == "ultimate":
        # s_prompt = f"You are an expert program repair system. You should carefully analyze problem descriptions and input/output specifications. You should reflect on previous failed repair attempts (the input, expected output, actual result and execution outcome of the test). You should make the analysis step by step. The output should be in json format."
        s_prompt = "You are an automated program repair tool."
    elif mode == "ultimate2":
        # s_prompt = "You are an expert program repair system. You should carefully analyze problem descriptions and input/output specifications. The buggy code that cannot be fixed will be translated to other programming languages for you to fix at each iteration. You should reflect on previous failed tests and provide the fixed code with the experience from historical failures."
        s_prompt = "you are an Automated Program Repair tool."
        if msg[0]["role"] != "system":
            system_msg = {"role": "system", "content": s_prompt}
            msg.insert(0, system_msg)
    else:
        s_prompt = None
    if not os.path.exists(file_path):
        # dt["prob_desc_sample_inputs"] = json.loads(dt["prob_desc_sample_inputs"])
        # dt["prob_desc_sample_outputs"] = json.loads(dt["prob_desc_sample_outputs"])
        if mode in ["testfailure", "ultimate"]:
            lm_io = prompt.apr_hist(dt)
        else:
            lm_io = prompt.apr(dt)
            print(f"lm_io: {lm_io}")

        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
        else:
            out = gen(lm_io[0], temperature, nsample, mode, llm, msg)
            print(f"out: {out}")

            # out = gen_request(s_prompt, lm_io[0], temperature, nsample, mode, msg)
            export_data = {"oai_response": out, "source_data": dt}
            print(f"export_data: {export_data}")

            open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")


def sanitize_code(code):
    prefixes = ["csharp", "cpp", "go", "javascript", "kotlin", "php", "python", "ruby", "rust", "c", "java"]
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

def load_json_files(dir):
    json_files = []
    files = os.listdir(dir)
    files.sort()
    for filename in files:
        if filename.endswith('.json'):
            file_path = os.path.join(dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    dt = data["source_data"]
                    code = data["oai_response"]["choices"][0]["message"]["content"]
                    code = sanitize_code(code)
                    dt["bug_source_code"] = code
                    dt["lang_cluster"] = dt["target_lang"]
                    json_files.append(dt)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    return json_files


def run(base_dir, num_proc, dry_run, nsample, nattempt, it, mode, temperature, dataset_path, llm):
    iter_dir = os.path.join(base_dir, f"iter_{it}")
    print(f"iter_dir: {iter_dir}")
    re_gen_dir = os.path.join(iter_dir, f"repair")
    print(f"re_gen_dir: {re_gen_dir}")
    if not os.path.exists(re_gen_dir):
        os.makedirs(re_gen_dir, exist_ok=True)

    unfixed_path = os.path.join(iter_dir, "unfixed.json")
    with open(unfixed_path, "r") as uf:
        unfixed_ids = json.load(uf).keys()

    print(f"unfixed_ids: {unfixed_ids}")

    if mode not in ["cmp"]:
        transed_dir = os.path.join(iter_dir, "trans")
        print(f"transed_dir: {transed_dir}")

        transed_dataset = load_json_files(transed_dir)
        print(f"transed_dataset: {transed_dataset}")

    elif mode == 'cmp':
        apr_dataset = datasets.load_from_disk(dataset_path)
        transed_dataset = apr_dataset.filter(lambda x: x["bug_code_uid"] in unfixed_ids)
    # elif mode == 'ultimate2':
    #     transed_dataset = get_last_incorrect_samples_cr(base_dir, it, unfixed_ids)

    if mode == "ultimate2":
        last_repair = load_last_repair(base_dir, it)
        last_tests = load_last_tests(base_dir, it)

    # temperature_list = np.linspace(0, 2, args.nsample)
    temperature_list = [temperature]
    # Sequential (no ProcessPoolExecutor) version
    errors = 0

    # Per-sample progress
    for idx, dt in tqdm.tqdm(
        enumerate(transed_dataset),
        total=len(transed_dataset),
        desc="Preparing samples lang",
    ):
        if mode in ["ultimate", "testfailure"]:
            dt = add_hist_testfailure(base_dir, dt, it)

        msg = None
        if mode == "ultimate2":
            msg = construct_conversation(base_dir, it, dt, last_repair, last_tests)

        # How many attempts to use for this run
        n_attempts = 1 if mode in ["ultimate2", "testfailure"] else nattempt

        for attempt in range(n_attempts):
            for temperature in temperature_list:
                try:
                    process_prompt(
                    dt,
                    temperature,
                    nsample,
                    re_gen_dir,
                    idx,
                    attempt,
                    mode,
                    llm,        # correct argument order
                    msg,
                    dry_run,
                    )
                except Exception as e:
                    errors += 1
                    print(f"Error occurred: {e}")
    if errors:
        print(f"Completed with {errors} error(s).")
    else:
        print("Completed without errors.")


    # with concurrent.futures.ThreadPoolExecutor(
    #     max_workers=int(num_proc)
    # ) as executor:
    #     futures = []
    #     for idx, dt in tqdm.tqdm(
    #         enumerate(transed_dataset),
    #         total=len(transed_dataset),
    #         desc=f"Preparing samples lang",
    #     ):
    #         if mode in ["ultimate", "testfailure"]:
    #             dt = add_hist_testfailure(base_dir, dt, it)
    #         msg = None
    #         if mode == "ultimate2":
    #             msg = construct_conversation(base_dir, it, dt, last_repair, last_tests)
    #             # with open("/root/TR/test/msg.txt", "a") as msg_file:
    #             #     msg_file.write(str(msg))
    #         if mode in ["ultimate2", "testfailure"]:
    #             nattempt = 1
    #         for attempt in range(nattempt):
    #             for temperature in temperature_list:
    #                 # process_prompt(dt, temperature, nsample, re_gen_dir, idx, attempt, mode, llm, msg, dry_run)

    #                 future = executor.submit(
    #                     process_prompt,
    #                     dt,
    #                     temperature,
    #                     nsample,
    #                     re_gen_dir,
    #                     idx,
    #                     attempt,
    #                     mode,
    #                     llm,
    #                     msg,
    #                     dry_run,
    #                 )
    #                 futures.append(future)

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
        "--it",
        default=1,
        type=int,
        help="Current iteration epoch of trans-repair.",
    )
    parser.add_argument(
        "--mode",
        default="vanilla",
        help="Repair mode.",
    )
    parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Set the temperature for the language model."
    )
    args = parser.parse_args()

    run(args.base_dir, args.num_proc, args.dry_run, args.nsample, args.nattempt, args.it, args.mode, args.temperature)

