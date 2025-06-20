from typing import List
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from math import comb

def codex_pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def create_problem_template(item: dict, include_buggy_code: bool = True) -> str:
    outcome_descriptions = {
        "COMPILATION_ERROR": "The buggy code fails to compile or run due to a syntax error.",
        "RUNTIME_ERROR": "The code compiles successfully but encounters an error during execution.",
        "MEMORY_LIMIT_EXCEEDED": "The code uses more memory than allowed.",
        "TIME_LIMIT_EXCEEDED": "The code takes too long to run.",
        "WRONG_ANSWER": "The code runs but produces incorrect output.",
        "PASSED": "The buggy code passes all tests (may indicate insufficient tests).",
    }
    exec_outcome = item["bug_exec_outcome"]
    description = outcome_descriptions.get(exec_outcome, "Unknown execution outcome.")

    template = f"""Problem description: {item['description']}

    Input format: {item['input_spec']}

    Output format: {item['output_spec']}

    Time limit: {item['time_limit']}

    Memory limit: {item['memory_limit']}

    A pre-run execution outcome of buggy source code: {exec_outcome} ({description})
    """
    if include_buggy_code:
        template = f"Buggy source code: {item['bug_source_code']}\n\n" + template

    return template


def generate_function(
        gen,
        item,
        model,
        strategy,
        cur_func_impl,
        reflections,
        is_first_reflection,
        prompting,
        feedback,
        inferred_specificaion,
        num_comps=1 ###change it
        ):
    problem_context = create_problem_template(item, include_buggy_code=False)

    if prompting == "scot":
        return gen.scot_func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=cur_func_impl,
            reflections=reflections,
            feedback=feedback,
            inferred_specificaion=inferred_specificaion,
            num_comps=num_comps
        )
    else:
        return gen.func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=cur_func_impl,
            reflections=reflections,
            feedback=feedback,
            inferred_specificaion=inferred_specificaion,
            num_comps=num_comps
        )


def run_single_item(
        item,
        i,
        exe,
        gen,
        model,
        pass_at_k,
        max_iters,
        prompting,
        verbose,
        infer_spec
    ):
    print_v = make_printv(verbose)
    is_first_reflection = True
    success_count = 0
    item["attempt_results"] = []
    iteration_pass_matrix = [[] for _ in range(max_iters)]

    for attempt_id in range(pass_at_k):
        is_solved = False
        reflections, implementations, test_feedback = [], [], []
        cur_func_impl = item["bug_source_code"]
        is_first_reflection = True
        cur_feedback = None

        try:

            if infer_spec:
                inferred_specificaion = gen.infer_specification(
                    problem_context=create_problem_template(item, False),
                    model=model,
                )
                        
            reflection = gen.first_reflection(
                problem_context=create_problem_template(item, False),
                inferred_specificaion=inferred_specificaion,
                func=item["bug_source_code"],
                model=model
            )
            reflections.append(reflection)

            samples = [(inp.replace(" ", "\n") + '\n', out)
                for inp, out in zip(item["sample_inputs"], item["sample_outputs"])]

            tests = gen.internal_tests(
                problem_context=create_problem_template(item, False),
                inferred_specificaion=inferred_specificaion,
                model=model,
                max_num_tests=7,
                samples=samples,
            )
            print(f"tests_i: {tests}")
            formatted_tests = [{"input": inp, "output": out} for inp, out in tests]

            # validated_tests = gen.validate_internal_tests(
            #     tests=tests,
            #     problem_context=create_problem_template(item, False),
            #     func=item["bug_source_code"],
            #     model=model,
            #     max_num_tests=5
            # )
            # print(f"validated_tests_i: {validated_tests}")

            func_impls = generate_function(
                gen,
                item,
                model,
                strategy="reflexion",
                cur_func_impl=item["bug_source_code"],
                problem_context=create_problem_template(item, False),
                inferred_specificaion=inferred_specificaion,
                reflections=reflections,
                is_first_reflection=is_first_reflection,
                prompting=prompting,
                num_comps=pass_at_k
            )

            is_first_reflection = False

            for cur_func_impl in func_impls:
                implementations.append(cur_func_impl)
                result = exe.execute(cur_func_impl, formatted_tests)
                is_passing = result["is_passing"]
                feedback = result["feedback"]
                test_feedback.append(feedback)
                iteration_pass_matrix[0].append(is_passing)

                if is_passing:
                    passed_all = exe.evaluate(
                        cur_func_impl,
                        item["unittest_cases"],
                        timeout=10
                    )
                    if passed_all:
                        success_count += 1
                        is_solved = True

            cur_iter = 1
            cur_feedback = feedback

            while cur_iter < max_iters:
                try:
                    reflection = gen.self_reflection(
                        problem_context=create_problem_template(item, False),
                        inferred_specificaion=inferred_specificaion,
                        cur_func_impl=cur_func_impl,
                        cur_feedback=cur_feedback,
                        model=model
                    )
                    reflections.append(reflection)
                    print(f"REFLECTION!!!!!!!!: {reflection}")
                    cur_func_impl = generate_function(
                        gen,
                        item,
                        model,
                        strategy="reflexion",
                        cur_func_impl=cur_func_impl,
                        problem_context=create_problem_template(item, False),
                        inferred_specificaion=inferred_specificaion,
                        reflections=reflections,
                        is_first_reflection=is_first_reflection,
                        prompting=prompting,
                        feedback=cur_feedback
                    )
                    implementations.append(cur_func_impl)

                    result = exe.execute(cur_func_impl, formatted_tests)
                    is_passing = result["is_passing"]
                    iteration_pass_matrix[cur_iter].append(is_passing)
                    cur_feedback = result["feedback"]
                    test_feedback.append(cur_feedback)

                    if is_passing or cur_iter == max_iters - 1:
                        is_passing = exe.evaluate(
                            cur_func_impl,
                            item["unittest_cases"],
                            timeout=10
                        )
                        if is_passing:
                            is_solved = True
                            success_count += 1
                        break

                    cur_iter += 1
                except Exception as e:
                    print(f"Error in iteration {cur_iter} for item {i}: {e}")
                    cur_iter += 1
                    continue

            # cur_pass += 1

        except Exception as e:
            print(f"Skipping item {i} due to error: {e}")
            break

    item["is_solved"] = is_solved
    item["reflections"] = reflections
    item["implementations"] = implementations
    item["test_feedback"] = test_feedback
    item["solution"] = cur_func_impl
    item["success_count"] = success_count
    item[f"pass@{pass_at_k}"] = codex_pass_at_k(pass_at_k, success_count, pass_at_k)

    for iter_idx, results in enumerate(iteration_pass_matrix):
        c = sum(results)
        item[f"pass@10_iter{iter_idx}"] = codex_pass_at_k(pass_at_k, c, 10)

    return item, item[f"pass@{pass_at_k}"]



def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    infer_spec: bool,
    is_leetcode: bool = False,
    model_path: str = None,
) -> None:
    prompting = "cot"
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name, model_path)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    total_success = resume_success_count(dataset)

    pass10_list = []
    for i, item in enumerate_resume(dataset, log_path):
        try:
            updated_item, pass10 = run_single_item(
                item, i, exe, gen, model, pass_at_k, max_iters, prompting, verbose, infer_spec
            )
            write_jsonl(log_path, [updated_item], append=True)
            pass10_list.append(updated_item[f"pass@{pass_at_k}"])
            print_v(f"completed {i+1}/{num_items}: pass@10 so far = {round(sum(pass10_list)/(i+1), 3)}")
        except Exception as e:
            print(f"Error processing item {i}: {e}. Continuing.")
            continue

    overall_pass10 = sum(pass10_list) / len(pass10_list)
    print(f"\n🟢 FINAL pass@10 across all {len(pass10_list)} bugs: {overall_pass10:.3f}")
