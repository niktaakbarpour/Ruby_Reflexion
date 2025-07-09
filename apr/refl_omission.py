from typing import List
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import json
from math import comb

def codex_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Exact formula from the OpenAI blog:
      pass@k = 1 − C(n − c, k) / C(n, k)
    where c is number of correct completions out of n.

    Edge‐cases:
      if c == 0: pass@k = 0
      if c > n − k: pass@k = 1 (every k‐subset must have ≥1 correct)
    """
    if c == 0:
        return 0.0
    if c > n - k:
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

    template = f"""Problem description: {item['prob_desc_description']}

    Input format: {item['prob_desc_input_spec']}

    Output format: {item['prob_desc_output_spec']}

    Time limit: {item['prob_desc_time_limit']}

    Memory limit: {item['prob_desc_memory_limit']}

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
        problem_context,
        reflections,
        is_first_reflection,
        prompting,
        feedback,
        # inferred_specificaion,
        num_comps=1 #n
        ):
    problem_context = create_problem_template(item, include_buggy_code=False)

    if prompting == "scot":
        out = gen.scot_func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=cur_func_impl,
            reflections=reflections,
            feedback=feedback,
            # inferred_specificaion=inferred_specificaion,
            num_comps=num_comps
        )
    else:
        out = gen.func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=cur_func_impl,
            reflections=reflections,
            feedback=feedback,
            # inferred_specificaion=inferred_specificaion,
            num_comps=num_comps
        )
    
    if isinstance(out, list):
        return out
    else:
        return [out]


def run_single_item(
        item,
        i,
        exe,
        gen,
        model,
        pass_at_k,
        n_completions,
        max_iters,
        prompting,
        verbose
    ):

    print("File refl_omission: ", f"GOING for I: {i}")


    print_v = make_printv(verbose)
    #num_success = 0
    success_count = 0
    iteration_pass_matrix = [[] for _ in range(max_iters)]
    iteration_unit_pass_matrix = [[] for _ in range(max_iters)]
    solved_iter = None
    ever_unit_ok = False
    #cur_pass = 0
    is_first_reflection = True
    is_solved = False
    reflections, implementations, test_feedback = [], [], []
    cur_func_impl = item["bug_source_code"]
    cur_feedback = None

    """
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
                func=item["bug_source_code"],
                inferred_specificaion=inferred_specificaion,
                model=model,
                max_num_tests=7,
                samples=samples,
            )
            print(f"tests_i: {tests}")

            cur_func_impl = generate_function(
                gen,
                item,
                model,
                strategy="refl_omission",
                cur_func_impl=item["bug_source_code"],
                problem_context=create_problem_template(item, False),
                inferred_specificaion=inferred_specificaion,
                is_first_reflection=is_first_reflection,
                prompting=prompting
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
                    cur_func_impl = generate_function(
                        gen,
                        item,
                        model,
                        strategy="refl_omission",
                        cur_func_impl=cur_func_impl,
                        is_first_reflection=is_first_reflection,
                        prompting=prompting,
                        feedback=cur_feedback,
                        problem_context=create_problem_template(item, False),
                        inferred_specificaion=inferred_specificaion,
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
    """
    inputs = json.loads(item["prob_desc_sample_inputs"])
    outputs = json.loads(item["prob_desc_sample_outputs"])

    samples = [{"input": inp + '\n', "output": out} for inp, out in zip(inputs, outputs)]

    tests = gen.internal_tests(
        problem_context=create_problem_template(item, False),
        # inferred_specificaion=inferred_specificaion,
        model=model,
        max_num_tests=7,
        samples=samples,
    )
    print("File refl_omission: ", f"tests_i: {tests}")
    formatted_tests = [
    {
        "input": ' '.join(inp.strip().split()) + '\n',
        "output": ''.join(out).strip() if isinstance(out, list) else out.strip()
    }
    for inp, out in tests
    ]

    print("File refl_omission: ",f"formatted_tests: {formatted_tests}")
    reflections.append("")
    func_impls = []
    batch_size = 1
    for b in range(0, n_completions, batch_size):
        impls = generate_function(
        gen,
        item,
        model,
        strategy="refl_omission",
        cur_func_impl=item["bug_source_code"],
        problem_context=create_problem_template(item, False),
        # inferred_specificaion=inferred_specificaion,
        reflections=reflections,
        is_first_reflection=is_first_reflection,
        prompting=prompting,
        feedback=None,
        num_comps=batch_size   
    )
        func_impls.extend(impls)
    
    is_first_reflection = False

    for cur_impl in func_impls:
        implementations.append(cur_impl)
        result = exe.execute(cur_impl, formatted_tests)
        is_passing = result["is_passing"]
        feedback = result["feedback"]
        print("File refl_omission: ",f"is_passing: {is_passing}")
        print("File refl_omission: ",f"feedback: {feedback}")
        test_feedback.append(feedback)
        iteration_pass_matrix[0].append(is_passing)
        cur_func_impl = cur_impl

        if isinstance(item["hidden_unit_tests"], str):
            item["hidden_unit_tests"] = json.loads(item["hidden_unit_tests"])

        unit_ok, unit_test_results = exe.evaluate(cur_func_impl, item["hidden_unit_tests"], timeout=10)
        ever_unit_ok = ever_unit_ok or unit_ok
        print("File refl_omission: ",f"unit_ok first: {unit_ok}")
        test_feedback.append(f"unit_tests_passed={unit_ok}")
        iteration_unit_pass_matrix[0].append(unit_ok)
        if unit_ok and is_passing:
            solved_iter = 0
            is_solved = True
            break

        cur_iter = 1
        cur_feedback = feedback

        while not is_solved and cur_iter < max_iters:
            print("File refl_omission: ",f"cur_iter: {cur_iter}")
            """reflection = gen.self_reflection(
                problem_context=create_problem_template(item, False),
                # inferred_specificaion=inferred_specificaion,
                func=cur_func_impl,
                feedback=cur_feedback,
                model=model,
                strategy="refl_omission"
            )
            reflections.append(reflection)"""
            # print("File refl_omission: ",f"REFLECTION!!!!!!!!: {reflection}")
            next_impls = generate_function(
                gen,
                item,
                model,
                strategy="refl_omission",
                cur_func_impl=cur_func_impl,
                problem_context=create_problem_template(item, False),
                # inferred_specificaion=inferred_specificaion,
                reflections=reflections,
                is_first_reflection=is_first_reflection,
                prompting=prompting,
                feedback=cur_feedback,
                num_comps=1
            )
            cur_func_impl = next_impls[0]
            implementations.append(cur_func_impl)

            result = exe.execute(cur_func_impl, formatted_tests)
            is_passing = result["is_passing"]
            iteration_pass_matrix[cur_iter].append(is_passing)
            cur_feedback = result["feedback"]
            test_feedback.append(cur_feedback)
            print("File refl_omission: ",f"is_passing2: {is_passing}")
            print("File refl_omission: ",f"feedback2: {cur_feedback}")

            if isinstance(item["hidden_unit_tests"], str):
                item["hidden_unit_tests"] = json.loads(item["hidden_unit_tests"])

            unit_ok, unit_test_results = exe.evaluate(cur_func_impl, item["hidden_unit_tests"], timeout=10)
            print("File refl_omission: ",f"unit_ok 2: {unit_ok}")
            ever_unit_ok = ever_unit_ok or unit_ok
            iteration_unit_pass_matrix[cur_iter].append(unit_ok)

            if is_passing or cur_iter == max_iters - 1:
                if unit_ok:
                    solved_iter = cur_iter
                    is_solved = True
                break

            cur_iter += 1

    if ever_unit_ok:
        success_count += 1
        is_solved = True

    item["is_solved"] = is_solved
    item["implementations"] = implementations
    item["test_feedback"] = test_feedback
    item["solution"] = cur_func_impl
    #item["inferred_specificaion"] = inferred_specificaion
    item["success_count"] = success_count
    item["solved_iteration"] = solved_iter
    item[f"pass@{pass_at_k}"] = codex_pass_at_k(n_completions, success_count, pass_at_k)
    item["final_unit_ok"] = unit_ok
    item["final_unit_test_results"] = unit_test_results
    
    print("File refl_omission: ",f"solved_iteration: {solved_iter}")
    print("File refl_omission: ",f"is_solvedF: {is_solved}")
    print("File refl_omission: ",f"success_count: {success_count}")
    print("File refl_omission: ",f"pass@{pass_at_k}: {codex_pass_at_k(n_completions, success_count, pass_at_k)}")
    
    for iter_idx, results in enumerate(iteration_pass_matrix):
        c = sum(results)
        item[f"pass@{pass_at_k}_iter{iter_idx}"] = codex_pass_at_k(n_completions, c, pass_at_k)
        print("File refl_omission: ",f"pass@{pass_at_k}_iter{iter_idx}: {codex_pass_at_k(n_completions, c, pass_at_k)}")

    for iter_idx, results in enumerate(iteration_unit_pass_matrix):
        c = sum(results)
        item[f"pass@{pass_at_k}_unit_iter{iter_idx}"] = codex_pass_at_k(n_completions, c, pass_at_k)
        print("File refl_omission: ",f"pass@{pass_at_k}_unit_iter{iter_idx}: {codex_pass_at_k(n_completions, c, pass_at_k)}")

    return item, item[f"pass@{pass_at_k}"]

def run_refl_omission(
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
    n_completions = 1 #n
    k = pass_at_k
    pass_list = []

    for i, item in enumerate_resume(dataset, log_path):
        try:
            updated_item, passk = run_single_item(
            item,
            i,
            exe,
            gen,
            model,
            k,
            n_completions,
            max_iters,
            prompting,
            verbose,
            # infer_spec
            )

            write_jsonl(log_path, [updated_item], append=True)
            pass_list.append(updated_item[f"pass@{pass_at_k}"])
            print_v(f"completed {i+1}/{num_items}: pass@{pass_at_k} so far = {round(sum(pass_list)/(i+1), 3)}")
        except Exception as e:
            print("File refl_omission: ",f"Error processing item {i}: {e}. Continuing.")
            continue

    if pass_list:
        overall_pass = sum(pass_list) / len(pass_list)
        print("File refl_omission: ",f"\n🟢 FINAL pass@{pass_at_k} across all {len(pass_list)} bugs: {overall_pass:.3f}")
    else:
        print("File refl_omission: ",f"\n⚠️  No bugs were processed, so pass@{pass_at_k} cannot be computed.")

# TEMP: triggering commit for review