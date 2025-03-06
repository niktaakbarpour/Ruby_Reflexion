from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List

SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
SIMPLE_CHAT_INSTRUCTION = "You are a Ruby programming assistant. You will be given a buggy code implementation and its docstring by the user. Write ONLY your full correct implementation in Ruby (DO NOT write example usage). In other words your task is automatic program repair."

def run_simple(
        dataset: List[dict],
        model_name: str,
        language: str,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        model_path:str = None
    ) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name, model_path)

    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = 0

    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        cur_func_impl = ""

        def create_template(json_data):
            template = f"""\
            Buggy source code: {json_data["bug_source_code"]}

            Problem description in textual format, math operations are written in LaTeX: {json_data["description"]}

            How and in what order the input will be given to the program? It also includes the data range, types, and sizes: {json_data["input_spec"]}

            How the outputs should be printed. Most of the time, the unit test results are matched with an exact string match or floating point comparison with a precision boundary: {json_data["output_spec"]}
            """
            return template

        modified_data = create_template(item)

        while cur_pass < pass_at_k:
            cur_func_impl = gen.func_impl(modified_data, model, "simple")
            assert isinstance(cur_func_impl, str)
            is_passing = exe.evaluate(cur_func_impl, item["unittest_cases"], timeout=10)
            if is_passing:
                is_solved = True
                num_success += 1
                break
            cur_pass += 1
        item["solution"] = cur_func_impl
        
        item["is_solved"] = is_solved
        write_jsonl(log_path, [item], append=True)

        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
