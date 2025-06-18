from typing import List
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory


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
        inferred_specificaion
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
            feedback=feedback
            inferred_specificaion=inferred_specificaion,
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
        verbose
    ):
    print_v = make_printv(verbose)
    num_success = 0
    cur_pass = 0
    is_first_reflection = True
    is_solved = False
    reflections, implementations, test_feedback = [], [], []
    cur_func_impl = ""

    while cur_pass < pass_at_k and not is_solved:
        try:

            inferred_specificaion = gen.infer_specification(
                problem_context=create_problem_template(item, False),
                model=model,
            )
                        
            samples = [(inp.replace(" ", "\n") + '\n', out)
                for inp, out in zip(item["sample_inputs"], item["sample_outputs"])]

            tests = gen.internal_tests(
                problem_context=create_problem_template(item, False),
                inferred_specificaion=inferred_specificaion,
                func=item["bug_source_code"],
                model=model,
                max_num_tests=7,
                samples=samples,
            )
            print(f"tests_i: {tests}")

            cur_func_impl = generate_function(
                gen,
                item,
                model,
                strategy="first_refl_omission",
                cur_func_impl=item["bug_source_code"],
                is_first_reflection=is_first_reflection,
                prompting=prompting,
                inferred_specificaion=inferred_specificaion,
                problem_context=create_problem_template(item, False),
            )
            implementations.append(cur_func_impl)
            is_first_reflection = False

            formatted_tests = [{"input": inp, "output": out} for inp, out in tests]
            result = exe.execute(cur_func_impl, formatted_tests)
            is_passing = result["is_passing"]
            feedback = result["feedback"]
            test_feedback.append(feedback)

            if is_passing:
                is_passing = exe.evaluate(cur_func_impl, item["unittest_cases"], timeout=10)
                if is_passing:
                    is_solved = True
                    num_success += 1
                    break

            cur_iter = 1
            cur_feedback = feedback

            while cur_iter < max_iters:
                try:
                    reflection = gen.self_reflection(
                        cur_func_impl,
                        cur_feedback,
                        model,
                        inferred_specificaion=inferred_specificaion,
                        problem_context=create_problem_template(item, False),
                    )
                    reflections.append(reflection)
                    print(f"REFLECTION!!!!!!!!: {reflection}")
                    cur_func_impl = generate_function(
                        gen,
                        item,
                        model,
                        strategy="first_refl_omission",
                        cur_func_impl=cur_func_impl,
                        reflections=reflections,
                        is_first_reflection=is_first_reflection,
                        prompting=prompting,
                        feedback=cur_feedback,
                        inferred_specificaion=inferred_specificaion,
                        problem_context=create_problem_template(item, False),
                    )
                    implementations.append(cur_func_impl)

                    result = exe.execute(cur_func_impl, formatted_tests)
                    is_passing = result["is_passing"]
                    cur_feedback = result["feedback"]
                    test_feedback.append(cur_feedback)

                    if is_passing or cur_iter == max_iters - 1:
                        is_passing = exe.evaluate(cur_func_impl, item["unittest_cases"], timeout=10)
                        if is_passing:
                            is_solved = True
                            num_success += 1
                        break

                    cur_iter += 1
                except Exception as e:
                    print(f"Error in iteration {cur_iter} for item {i}: {e}")
                    cur_iter += 1
                    continue

            cur_pass += 1

        except Exception as e:
            print(f"Skipping item {i} due to error: {e}")
            break

    item["is_solved"] = is_solved
    item["reflections"] = reflections
    item["implementations"] = implementations
    item["test_feedback"] = test_feedback
    item["solution"] = cur_func_impl

    return item, num_success


def run_first_refl_omission(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    model_path: str = None
) -> None:
    prompting = "cot"
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name, model_path)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    total_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path):
        try:
            updated_item, num_success = run_single_item(
                item, i, exe, gen, model, pass_at_k, max_iters, prompting, verbose
            )
            write_jsonl(log_path, [updated_item], append=True)
            total_success += num_success

            print_v(f"completed {i+1}/{num_items}: acc = {round(total_success/(i+1), 2)}")
        except Exception as e:
            print(f"Error processing item {i}: {e}. Continuing with next item.")
            continue
