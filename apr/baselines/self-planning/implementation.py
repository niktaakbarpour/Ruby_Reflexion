import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Adjust if needed
sys.path.append(os.path.dirname(project_root))
import time
import tqdm
import json
import openai
import requests
# from openai import OpenAI
import argparse
import datasets
import concurrent
import numpy as np
from promptsource.templates import Template
from middleware.repair_retrieval import add_hist, construct_conversation, add_hist_testfailure
from middleware.history import load_last_repair, load_last_tests, get_last_incorrect_samples_cr
from middleware import prompt
from middleware.deepseek_local import Message
from middleware.deepseek_local import DeepSeekCoder
import atexit
import time
import psutil
import pynvml


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


def gen(prompt_text, temperature, nsample, llm):
    cnt = 0
    while cnt < 999:
        print(cnt)
        try:
            print(cnt)
            # Prepare messages
            messages = [
                Message(role="system", content=prompt.PROMPTS["system"]),
                Message(role="user", content=prompt_text),
            ]

            # Generate response from local model
            c = llm.generate_chat(messages, temperature=temperature, num_comps=nsample)

            break
        except Exception as e:
            cnt += 1
            print(f"cnt error: {cnt}")
            time.sleep(5)
            print(f"Gen Error: {e}")
    else:
        return None

    # Add the prompt field to mimic OpenAI's structure
    c["prompt"] = prompt_text
    return c




def process_prompt(dt, nsample, output_dir, index, temperature, llm, dry_run=0):
    language = dt["lang_cluster"]
    uid = dt["bug_code_uid"]
    file_path = os.path.join(output_dir, f"{index}_{uid}_{language}.json")

    if not os.path.exists(file_path):
        lm_io = prompt.imp(dt)
        print(lm_io)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
        else:
            print("out")
            out = gen(lm_io[0], temperature, nsample, llm)
            print(f"Shows tokens for this generation: {out['usage']}")
            print(f"Shows total tokens used so far: {llm.get_total_tokens()}")
            # out = gen_request(s_prompt, lm_io[0], temperature, nsample, mode, msg)
            export_data = {"oai_response": out, "source_data": dt}
            open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")

def sanitize_plan(plan):
    FLAG = True
    while FLAG == True:
        FLAG = False
        if plan.startswith("[Plan]"):
            FLAG = True
            plan = plan.replace("[Plan]", "", 1)
        elif plan.startswith("[Plan 6]"):
            FLAG = True
            plan = plan.replace("[Plan 6]", "", 1)
    return plan

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
                    plan = data["oai_response"]["choices"][0]["message"]["content"]
                    plan = sanitize_plan(plan)
                    dt["plan"] = plan
                    json_files.append(dt)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    return json_files


def run(base_dir, num_proc, dry_run, nsample, temperature):

    def log_resources(prefix=""):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        pynvml.nvmlShutdown()
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.Process().memory_info().rss / 1024**2
        print(f"{prefix} GPU memory: {gpu_mem:.2f} MB | CPU usage: {cpu}% | RAM usage: {ram:.2f} MB")

    start_time = time.time()
    log_resources("[START]")
    
    imp_dir = os.path.join(base_dir, f"imp")
    if not os.path.exists(imp_dir):
        os.makedirs(imp_dir, exist_ok=True)


    plan_dir = os.path.join(base_dir, "plans")
    apr_dataset = load_json_files(plan_dir)

    llm = DeepSeekCoder("")
    print("model loaded")
    atexit.register(lambda: print(f"\n==== Final Token Usage ====\n{llm.get_total_tokens()}"))


    try:
        for idx, dt in tqdm.tqdm(
            enumerate(apr_dataset),
            total=len(apr_dataset),
            desc=f"Processing samples"
        ):
            try:
                print("in try")
                process_prompt(dt, nsample, imp_dir, idx, temperature, llm, dry_run)

            except Exception as e:
                print(f"Error occurred in task {idx}: {e}")

    finally:
        end_time = time.time()
        log_resources("[END]")
        print(f"Total time: {end_time - start_time:.2f} seconds")



    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=int(num_proc)
    # ) as executor:
    #     futures = []
    #     for idx, dt in tqdm.tqdm(
    #         enumerate(apr_dataset),
    #         total=len(apr_dataset),
    #         desc=f"Preparing samples lang",
    #     ):
    #         future = executor.submit(
    #             process_prompt,
    #             dt,
    #             nsample,
    #             imp_dir,
    #             idx,
    #             dry_run,
    #         )
    #         futures.append(future)

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
        default="",
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
    parser.add_argument(
        "--temperature",
        default=0.8,
        type=float,
        help="temperature.",
    )
    args = parser.parse_args()

    run(args.base_dir, args.num_proc, args.dry_run, args.nsample, args.temperature)
