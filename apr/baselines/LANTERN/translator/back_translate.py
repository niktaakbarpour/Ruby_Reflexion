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
from middleware import prompt
import shutil
from middleware.deepseek_local import Message

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
    {
      "choices": [
        {"index": i, "message": {"role":"assistant","content": ...}, "finish_reason":"stop"},
        ...
      ]
    }
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
            # Already OpenAI-ish: ensure minimal fields and return
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

    # enforce nsample
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
    """
    Generate a response using local DeepSeekCoder, normalized to OpenAI ChatCompletion style.
    """
    cnt = 0
    while cnt < 999:
        try:
            messages = [Message(role="user", content=prompt_text)]
            raw = llm.generate_chat(messages, temperature=temperature, num_comps=nsample)

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
#     """
#     Generate a response using local DeepSeekCoder, mimicking OpenAI ChatCompletion style.
#     """
#     cnt = 0
#     while cnt < 999:
#         try:
#             messages = [Message(role="user", content=prompt_text)]

#             # Use llm.generate_chat to handle prompt construction & generation
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

#     # Add prompt field to mimic OpenAI response
#     c["prompt"] = prompt_text
#     return c

def gen_request(prompt_text, temperature, nsample, llm):
    # alias to gen, returns data/type format
    res = gen(prompt_text, temperature, nsample, llm)
    if res is None:
        return None
    return {"data": [{"content": res['choices'][0]['message']['content'], "type": "text"}],
            "prompt": prompt_text}


def process_prompt(dt, temperature, back_trans_dir, llm, dry_run=0):
    file_name = dt["transed_file"]
    file_path = os.path.join(back_trans_dir, f"{file_name}.json")
    if not os.path.exists(file_path):
        lm_io = prompt.back_trans(dt)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
        else:
            out = gen(lm_io[0], temperature, 1, llm)
            # out = gen_request(lm_io[0], temperature, 1)
            export_data = {"oai_response": out, "source_data": dt}
            open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")

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

def load_json_files(dir):
    json_files = []
    for filename in os.listdir(dir):
        if filename.endswith('.json'):
            file_path = os.path.join(dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    dt = data["source_data"]
                    code = data["oai_response"]["choices"][0]["message"]["content"]
                    is_json = "```json" in code
                    code = sanitize_code(code)
                    if is_json:
                        res_data = json.loads(code)
                        code = res_data["fixed code"]
                    dt["bug_source_code"] = code
                    dt["lang_cluster"] = dt["source_lang"]
                    dt["transed_file"] = filename.split(".")[0]
                    json_files.append(dt)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    return json_files


def run(base_dir, num_proc, dry_run, it, mode, llm):
    iter_dir = os.path.join(base_dir, f"iter_{it}")
    back_trans_dir = os.path.join(iter_dir, "back_trans")
    if not os.path.exists(back_trans_dir):
        os.makedirs(back_trans_dir, exist_ok=True)
    # for chatrepair
    if mode in ['ultimate2', "testfailure"]:
        print('copying...')
        source_dir = f"{base_dir}/iter_{it}/repair"
        destination_dir = f"{base_dir}/iter_{it}/back_trans/"

        # Iterate over all files in the source directory
        for file_name in os.listdir(source_dir):
            full_file_name = os.path.join(source_dir, file_name)
            # Check if it's a file (not a directory)
            if os.path.isfile(full_file_name):
                # Copy the file to the destination directory
                shutil.copy(full_file_name, destination_dir)
        return

    # Load trans-repair outputs
    transed = load_json_files(os.path.join(iter_dir, 'repair'))
    temps = [0.3157894736842105]

    for dt in transed:
        for temp in temps:
            try:
                process_prompt(dt, temp, back_trans_dir, llm, dry_run)
            except Exception as e:
                print(f"Error occurred while back-translating: {e}")


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
    args = parser.parse_args()

    run(args.base_dir, args.num_proc, args.dry_run, args.it)

