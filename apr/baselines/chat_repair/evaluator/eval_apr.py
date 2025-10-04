import os
import json
import tqdm
import jsonlines
import datasets
import concurrent.futures
from dataclasses import dataclass, field
import itertools
import argparse
import requests
from typing import List, Optional, Union, Tuple
from enum import Enum
from multiprocessing import Pool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ExecOutcome(Enum):
    PASSED = "PASSED"  # code executes and output matches expected output
    WRONG_ANSWER = (
        "WRONG_ANSWER"  # code executes and output does NOT matches expected output
    )
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"  # code executes and didn't exit in time, output is ignored in this case
    RUNTIME_ERROR = "RUNTIME_ERROR"  # code failed to execute (crashed)
    COMPILATION_ERROR = "COMPILATION_ERROR"  # code failed to compile
    MEMORY_LIMIT_EXCEEDED = (
        "MEMORY_LIMIT_EXCEEDED"  # code exceeded memory limit during execution
    )


@dataclass
class ExtendedUnittest:
    input: str
    output: List[str] = field(default_factory=list)
    result: Optional[str] = None
    exec_outcome: Optional[ExecOutcome] = None

    def json(self):
        _json = self.__dict__
        if self.exec_outcome is not None:
            _json["exec_outcome"] = self.exec_outcome.name

        return _json

    @classmethod
    def from_json(cls, _json):
        return cls(
            input=_json.get("input", ""),
            output=_json.get("output", list()),
            result=_json.get("result", None),
            exec_outcome=_json.get("exec_outcome", None),
        )


class EmptyValueError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EmptyUnittestError(EmptyValueError):
    pass


class EmptyLanguageError(EmptyValueError):
    pass


class EmptySourceCodeError(EmptyValueError):
    pass


class APICommunication:
    _session: requests.Session

    def __init__(self, server_url: str = "http://127.0.0.1:5000",
                 timeout: Tuple[float, float] = (2.0, 600.0),
                 max_pool: int = 32):
        self._session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.2,
                      status_forcelist=(502, 503, 504),
                      allowed_methods=frozenset(["GET", "POST"]))
        adapter = HTTPAdapter(pool_connections=max_pool, pool_maxsize=max_pool, max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._timeout = timeout
        self.execute_code_url = f"{server_url}/api/execute_code"
        self.get_runtimes_url = f"{server_url}/api/all_runtimes"

    def __enter__(self):   # <-- outdented to class level
        return self

    def __exit__(self, *args):
        self._session.close()

    def _post_json(self, url: str, payload: dict):
        r = self._session.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self._timeout
        )
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            raise RuntimeError(f"Non-JSON response from {url}: {r.text[:500]}") from e

    def get_runtimes(self):
        r = self._session.get(self.get_runtimes_url, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def execute_code(
        self,
        language: str,
        source_code: str,
        unittests: List[dict],
        limits: Optional[dict] = None,
        block_network: bool = True,
        stop_on_first_fail: bool = True,
        use_sanitizer: bool = False,
        compiler_program_name: Optional[str] = None,
        compiler_flags: Optional[str] = None,
        interpreter_cmd: Optional[str] = None,
        interpreter_flags: Optional[str] = None,
        sample_id: Optional[int] = None,
        task_id: Union[str, int, None] = None,
    ) -> Tuple[List[ExtendedUnittest], Optional[int], Union[str, int, None]]:
        if language is None:
            raise EmptyLanguageError
        if source_code is None:
            raise EmptySourceCodeError
        if unittests is None or len(unittests) == 0:
            raise EmptyUnittestError

        request_body = dict(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else None,
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer,
        )
        json_response = self._post_json(self.execute_code_url, request_body)
        if "data" not in json_response:
            return json_response, sample_id, task_id
        return (json_response["data"], sample_id, task_id)



def get_idx(file_name):
    return int(file_name.split(".json")[0].split("_")[0])


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


def fix_uts(uts):
    uts_fx = []
    for ut in uts:
        uts_fx.append(
            {
                "input": ut["input"],
                "output": ut["output"],
            }
        )
    return uts_fx


# def process(args):
#     sample, execeval = args
#     src_uid = sample["source_data"]["src_uid"]
#     unit_tests = json.loads(sample["source_data"]["hidden_unit_tests"])
#     compiler = LANG_CLUSTER_TO_LANG_COMPILER[sample["source_data"]["lang_cluster"]]
#     sample["unit_test_results"] = []
#     # Ensure 'choices' exists in oai_response
#     if "choices" not in sample["oai_response"]:
#         # Fallback: convert from "data" if needed
#         if "data" in sample["oai_response"]:
#             sample["oai_response"]["choices"] = [
#                 {"message": {"content": item["content"]}} for item in sample["oai_response"]["data"]
#             ]
#         else:
#             print(f"❌ sample[{src_uid}] is missing both 'choices' and 'data'")
#             return  # or raise an exception

#     for choice in sample["oai_response"]["choices"]:
#         code = choice["message"]["content"]
#         code = sanitize_code(code)
#         unit_test_results, _, _ = execeval.execute_code(
#             compiler,
#             code,
#             fix_uts(unit_tests),
#             task_id=src_uid,
#             stop_on_first_fail=False # False for check
#         )
#         # print(file, code, [e['exec_outcome'] for e in unit_test_results])
#         sample["unit_test_results"].append(unit_test_results)
#     return sample


def process(sample, server_url: str):
    src_uid = sample["source_data"]["src_uid"]
    unit_tests = json.loads(sample["source_data"]["hidden_unit_tests"])
    compiler = LANG_CLUSTER_TO_LANG_COMPILER[sample["source_data"]["lang_cluster"]]
    sample["unit_test_results"] = []

    # Ensure 'choices' exists in oai_response
    if "choices" not in sample["oai_response"]:
        if "data" in sample["oai_response"]:
            sample["oai_response"]["choices"] = [
                {"message": {"content": item["content"]}} for item in sample["oai_response"]["data"]
            ]
        else:
            print(f"❌ sample[{src_uid}] is missing both 'choices' and 'data'")
            return

    # Per-thread API client with bigger pool & timeouts
    with APICommunication(server_url=server_url) as execeval:
        for choice in sample["oai_response"]["choices"]:
            code = sanitize_code(choice["message"]["content"])
            unit_test_results, _, _ = execeval.execute_code(
                compiler,
                code,
                fix_uts(unit_tests),
                task_id=src_uid,
                stop_on_first_fail=False  # False for check
            )
            sample["unit_test_results"].append(unit_test_results)
    return sample


LANG_CLUSTER_TO_LANG_COMPILER = {
    "C": "GNU C11",
    "C#": "Mono C#",
    "C++": "GNU C++17",
    "Go": "Go",
    "Java": "Java 17",
    "Javascript": "Node.js",
    "Kotlin": "Kotlin 1.4",
    "PHP": "PHP",
    "Python": "PyPy 3",
    "Ruby": "Ruby 3",
    "Rust": "Rust 2018",
}


def run(base_dir, it, mode, llm=None):
    # path = f'{os.environ["DUMP_FOLDER"]}/oai/apr_n_sample_20/'
    
    if it:
        path = os.path.join(base_dir, f"iter_{it}")
    else:
        path = base_dir

    for k, debug_compiler in LANG_CLUSTER_TO_LANG_COMPILER.items():
        if it:
            output_path = os.path.join(path, "eval")
        else:
            output_path = os.path.join(path, "eval_apr_val_execeval")
        # check the repair before back-trans
        if mode == 'check':
            output_path = os.path.join(path, "eval_check")
        elif mode == 'check_trans':
            output_path = os.path.join(path, "eval_trans")
        elif mode == 'check_original':
            output_path = os.path.join(path, "eval_original")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{debug_compiler}.jsonl")
        with jsonlines.open(output_file, "w") as jwp:
            # === choose the directory ===
            if it and mode == "vanilla":
                path_to_eval = os.path.join(path, "back_trans")
            elif it and mode in ["check_original", "check_trans"]:
                path_to_eval = os.path.join(path, "trans")
            elif mode == "self_planning":
                path_to_eval = os.path.join(path, "imp")
            else:
                path_to_eval = os.path.join(path, "repair")
            print(f"path_to_eval: {path_to_eval}")

            # list and filter filenames once
            all_entries = sorted(os.listdir(path_to_eval))
            files = [fname for fname in all_entries if "C++" in fname]
            print(f"all entries: {all_entries}")
            print(f"filtered files (contain 'C++'): {files}")

            # === build the batch ===
            all_samples = []
            for file in files:
                full_path = os.path.join(path_to_eval, file)
                print(f"full_path: {full_path}")

                # skip directories
                if os.path.isdir(full_path):
                    print(f"Skipping directory: {full_path}")
                    continue

                # load JSON sample
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        sample = json.load(f)
                except Exception as e:
                    print(f"Skipping {full_path}: failed to load JSON ({e})")
                    continue

                if mode == "check_trans":
                    sample["source_data"]["lang_cluster"] = sample["source_data"]["target_lang"]
                if mode == "check_original":
                    sample["oai_response"]["choices"][0]["message"]["content"] = sample["source_data"]["bug_source_code"]

                if sample["source_data"]["lang_cluster"] not in LANG_CLUSTER_TO_LANG_COMPILER:
                    continue
                compiler = LANG_CLUSTER_TO_LANG_COMPILER[sample["source_data"]["lang_cluster"]]
                if compiler != debug_compiler:
                    continue

                all_samples.append(sample)

            # === process sequentially ===
            server_url = os.environ.get("EXEC_ENGINE_URL", "http://127.0.0.1:5000")

            # ---- PRE-FLIGHT: verify engine is reachable (once per compiler batch) ----
            try:
                with APICommunication(server_url=server_url) as execeval_check:
                    execeval_check.get_runtimes()  # cheap GET
            except Exception as e:
                raise RuntimeError(
                    f"Engine not reachable at {server_url}. Aborting this batch ({debug_compiler})."
                ) from e

            for sample in tqdm.tqdm(
                all_samples,
                total=len(all_samples),
                desc=f"{debug_compiler}",
            ):
                try:
                    out = process(sample, server_url)
                    if out is not None:
                        jwp.write(out)
                except Exception as emsg:
                    print(f"Exception msg: {emsg}")
                    continue

        # with jsonlines.open(output_file, "w") as jwp:
        #     # === choose the directory ===
        #     if it and mode == "vanilla":
        #         path_to_eval = os.path.join(path, "back_trans")
        #     elif it and mode in ['check_original', 'check_trans']:
        #         path_to_eval = os.path.join(path, 'trans')
        #     elif mode == 'self_planning':
        #         path_to_eval = os.path.join(path, 'imp')
        #     else:
        #         path_to_eval = os.path.join(path, "repair")
        #         print(f"path_to_eval: {path_to_eval}")

        #     files = sorted(os.listdir(path_to_eval))
        #     print(f"files: {files}")

        #     # === build the batch ===
        #     all_samples = []
        #     for file in files:
        #         full_path = os.path.join(path_to_eval, file)
        #         print(f"full_path: {full_path}")
        #         # keep only the files that have "C#" in their name
        #         filtered_full_path = [
        #             os.path.join(path_to_eval, file)
        #             for file in files
        #             if "C#" in file
        #         ]

        #         print("Filtered paths:", filtered_full_path)

        #         if os.path.isdir(filtered_full_path):
        #             continue

        #         try:
        #             with open(filtered_full_path, "r", encoding="utf-8") as f:
        #                 sample = json.load(f)
        #         except Exception as e:
        #             print(f"Skipping {filtered_full_path}: failed to load JSON ({e})")
        #             continue

        #         if mode == 'check_trans':
        #             sample['source_data']['lang_cluster'] = sample['source_data']['target_lang']
        #         if mode == 'check_original':
        #             sample['oai_response']['choices'][0]['message']['content'] = sample['source_data']['bug_source_code']

        #         if sample["source_data"]["lang_cluster"] not in LANG_CLUSTER_TO_LANG_COMPILER:
        #             continue
        #         compiler = LANG_CLUSTER_TO_LANG_COMPILER[sample["source_data"]["lang_cluster"]]
        #         if compiler != debug_compiler:
        #             continue

        #         all_samples.append(sample)

        #     # === process sequentially ===
        #     server_url = os.environ.get("EXEC_ENGINE_URL", "http://127.0.0.1:5000")

        #     # ---- PRE-FLIGHT: verify engine is reachable (once per compiler batch) ----
        #     try:
        #         with APICommunication(server_url=server_url) as execeval_check:
        #             execeval_check.get_runtimes()  # cheap GET
        #     except Exception as e:
        #         # Stop early: otherwise every sample would fail with connection refused
        #         raise RuntimeError(f"Engine not reachable at {server_url}. Aborting this batch ({debug_compiler}).") from e


        #     for sample in tqdm.tqdm(
        #         all_samples,
        #         total=len(all_samples),
        #         desc=f"{debug_compiler}",
        #     ):
        #         try:
        #             out = process(sample, server_url)
        #             if out is not None:
        #                 jwp.write(out)
        #         except Exception as emsg:
        #             print(f"Exception msg: {emsg}")
        #             # keep going
        #             continue

            # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as thread_executor:
            #     if it and mode == "vanilla":
            #         path_to_eval = os.path.join(path, "back_trans")
            #     elif it and mode in ['check_original', 'check_trans']:
            #         path_to_eval = os.path.join(path, 'trans')
            #     elif mode == 'self_planning':
            #         path_to_eval = os.path.join(path, 'imp')
            #     else:
            #         path_to_eval = os.path.join(path, "repair")
            #         print (f"path_to_eval: {path_to_eval}")

            #     files = sorted(os.listdir(path_to_eval))
            #     print (f"files: {files}")

            #     # ---- build the batch WHILE the executor is still open ----
            #     all_samples = []
            #     for file in files:
            #         full_path = os.path.join(path_to_eval, file)
            #         print (f"full_path: {full_path}")

            #         if os.path.isdir(full_path):
            #             continue
            #         sample = json.load(open(full_path))
            #         if mode == 'check_trans':
            #             sample['source_data']['lang_cluster'] = sample['source_data']['target_lang']
            #         if mode == 'check_original':
            #             sample['oai_response']['choices'][0]['message']['content'] = sample['source_data']['bug_source_code']
            #         if sample["source_data"]["lang_cluster"] not in LANG_CLUSTER_TO_LANG_COMPILER:
            #             continue
            #         compiler = LANG_CLUSTER_TO_LANG_COMPILER[sample["source_data"]["lang_cluster"]]
            #         if compiler != debug_compiler:
            #             continue
            #         all_samples.append(sample)

            #     # ---- submit & consume futures INSIDE the executor context ----
            #     server_url = os.environ.get("EXEC_ENGINE_URL", "http://127.0.0.1:5000")
            #     futures = [thread_executor.submit(process, sample, server_url) for sample in all_samples]

            #     for _out in tqdm.tqdm(
            #         concurrent.futures.as_completed(futures),
            #         total=len(all_samples),
            #         desc=f"{debug_compiler}",
            #     ):
            #         try:
            #             __out = _out.result()
            #             if __out is not None:
            #                 jwp.write(__out)
            #         except Exception as emsg:
            #             print(f"Exception msg: {emsg}")
            #             pass





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="dumped/oai/apr_n_sample_20",
        help="Path to the trans-repair base directory.",
    )
    parser.add_argument(
        "--it",
        default=0,
        type=int,
        help="Current iteration epoch of trans-repair.",
    )
    parser.add_argument(
        "--mode",
        default="vanilla",
        help="Repair mode.",
    )
    arg = parser.parse_args()
    run(arg.base_dir, arg.it, arg.mode)

