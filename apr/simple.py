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

        def create_problem_template(item: dict, include_buggy_code: bool = True) -> str:

            template = f"""Problem description: {item['description']}

            Input format: {item['input_spec']}

            Output format: {item['output_spec']}
            """
            if include_buggy_code:
                template = f"Buggy source code: {item['bug_source_code']}\n\n" + template

            return template

        problem_context = create_problem_template(item, include_buggy_code=False)

        while cur_pass < pass_at_k:
            cur_func_impl = gen.func_impl(
                problem_context=problem_context,
                model=model,
                strategy="simple",
                prev_func_impl=cur_func_impl
            )
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
