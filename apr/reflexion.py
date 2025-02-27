from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
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
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        def add_function_container(code: str) -> str:
            return f"def run_buggy()\n{code}\nend"
        
        def create_template(json_data):
            template = f"""\
        Buggy source code: {add_function_container(json_data["bug_source_code"])}

        Problem description in textual format, math operations are written in LaTeX: {json_data["description"]}

        How and in what order the input will be given to the program? It also includes the data range, types, and sizes: {json_data["input_spec"]}

        How the outputs should be printed. Most of the time, the unit test results are matched with an exact string match or floating point comparison with a precision boundary: {json_data["output_spec"]}

        A sample input for the code that is expected to solve the problem described in the description: {json_data["sample_inputs"]}

        The expected output for the sample input that is expected to solve the problem described in the description: {json_data["sample_outputs"]}

        Explanation of sample inputs & sample outputs: {json_data["notes"]}
        """
            return template

        modified_data = create_template(item)
        while cur_pass < pass_at_k and not is_solved:
            print(f"cur_pass: {cur_pass}")
            tests_i = gen.internal_tests(modified_data, model, 5)

            print(f"tests_i *******: {tests_i}")

            # first attempt
            cur_func_impl = gen.func_impl(modified_data, model, "simple")
            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            result = exe.execute(cur_func_impl, tests_i)
            is_passing = result["is_passing"]
            feedback = result["feedback"]

            test_feedback.append(feedback)

            # if solved, exit early
            if is_passing:
                print("I'm here.6")
                is_passing = exe.evaluate(cur_func_impl, item["unittest_cases"], timeout=10)
                print(f"is_passing1: {is_passing}")
                is_solved = is_passing
                num_success += int(is_passing)
                print(f"num_success1: {num_success}")
                break

            # use self-reflection to iteratively improve
            cur_iter = 1
            cur_feedback = feedback
            print("I'm here.7")
            print(f"cur_iter: {cur_iter}")
            print(f"max_iter: {max_iters}")
            while cur_iter < max_iters:
                # get self-reflection
                reflection = gen.self_reflection(
                    cur_func_impl, cur_feedback, model)
                reflections += [reflection]

                print("I'm here.8")

                # apply self-reflection in the next attempt
                cur_func_impl = gen.func_impl(
                    func_sig=item["bug_source_code"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # check if all internal unit tests pass
                result = exe.execute(cur_func_impl, tests_i)
                # is_passing, cur_feedback, _ = exe.execute(
                #     cur_func_impl, tests_i)
                is_passing = result["is_passing"]
                cur_feedback = result["feedback"]
                test_feedback.append(cur_feedback)

                # if solved, check if it passes the real tests, exit early
                if is_passing or cur_iter == max_iters - 1:
                    print("I'm here.9")
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    print(f"is_passing2: {is_passing}")
                    if is_passing:
                        print("I'm here.10")
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                        print(f"num_success2: {num_success}")
                    break
                print("I'm here.11")
                cur_iter += 1
            print("I'm here.12")
            cur_pass += 1

        print("I'm here.13")

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl

        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
