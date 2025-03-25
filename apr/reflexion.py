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
    # dataset = [item for item in dataset if item.get("difficulty") == 800]

    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name, model_path)

    print_v = make_printv(verbose)

    num_items = len(dataset)

    num_success = resume_success_count(dataset)
    
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_first_reflection = True
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""

        def create_template(json_data, include_buggy_code=True):
            exec_outcome = json_data["bug_exec_outcome"]

            outcome_descriptions = {
                "COMPILATION_ERROR": "The buggy code fails to compile or run due to a syntax error.",
                "RUNTIME_ERROR": "The code compiles successfully but encounters an error during execution, such as division by zero or assertion failures.",
                "MEMORY_LIMIT_EXCEEDED": "The code uses more memory than the allowed limit and is terminated.",
                "TIME_LIMIT_EXCEEDED": "The code takes longer than the allowed execution time and is terminated.",
                "WRONG_ANSWER": "The code compiles and runs but does not produce the correct output.",
                "PASSED": "The buggy code unexpectedly passes all unit tests, meaning it might not be buggy or the tests are insufficient.",
            }

            description = outcome_descriptions.get(exec_outcome, "Unknown execution outcome.")

            template = ""

            if include_buggy_code:
                template += f"Buggy source code: {json_data['bug_source_code']}\n\n"

            template += f"""Problem description: {json_data["description"]}

Input format: {json_data["input_spec"]}

Output format: {json_data["output_spec"]}

A pre-run execution outcome of buggy source code: {exec_outcome} ({description})
        """

            return template

        
        # A sample input for the code that is expected to solve the problem described in the description: {json_data["sample_inputs"]}

        # The expected output for the sample input that is expected to solve the problem described in the description: {json_data["sample_outputs"]}

        # Explanation of sample inputs & sample outputs: {json_data["notes"]}

        modified_data = create_template(item, True)
        while cur_pass < pass_at_k and not is_solved:
            reflection = gen.first_reflection(problem_context = create_template(item, False),
                                              prev_func_impl = item["bug_source_code"],
                                              model = model)
            reflections += [reflection]
            tests_i = gen.internal_tests(problem_context = create_template(item, False),
                                              prev_func_impl = item["bug_source_code"],
                                              model = model,
                                              max_num_tests = 5)
            print(f"tests_i: {tests_i}")

            # first attempt
            cur_func_impl = gen.func_impl(problem_context = create_template(item, False),
                                          model = model,
                                          strategy = "reflexion",
                                          is_first_reflection = is_first_reflection,
                                          prev_func_impl = item["bug_source_code"],
                                          self_reflection = reflection)
            is_first_reflection = False
            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)

            # Convert (input_str, output_list) tuples into {"input": ..., "output": ...} dicts
            formatted_tests = [{"input": inp, "output": out} for inp, out in tests_i]
            # Now pass the correctly formatted tests to execute
            result = exe.execute(cur_func_impl, formatted_tests)

            # result = exe.execute(cur_func_impl, tests_i)
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
                    problem_context=create_template(item, False),
                    model=model,
                    strategy="reflexion",
                    is_first_reflection=is_first_reflection,
                    prev_func_impl=cur_func_impl,
                    self_reflection=reflection,
                    feedback=cur_feedback,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Convert (input_str, output_list) tuples into {"input": ..., "output": ...} dicts
                formatted_tests = [{"input": inp, "output": out} for inp, out in tests_i]
                # Now pass the correctly formatted tests to execute
                result = exe.execute(cur_func_impl, formatted_tests)

                # is_passing, cur_feedback, _ = exe.execute(
                #     cur_func_impl, tests_i)
                is_passing = result["is_passing"]
                cur_feedback = result["feedback"]
                test_feedback.append(cur_feedback)

                # if solved, check if it passes the real tests, exit early
                if is_passing or cur_iter == max_iters - 1:
                    print("I'm here.9")
                    is_passing = exe.evaluate(cur_func_impl, item["unittest_cases"], timeout=10)
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
