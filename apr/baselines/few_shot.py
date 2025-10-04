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

    print("File few-shot: ", f"GOING for I: {i}")

    print(f"item: {item}")
    print(f"model: {model}")
    print(f"prompting: {prompting}")
    print_v = make_printv(verbose)
    #num_success = 0
    success_count = 0
    iteration_unit_pass_matrix = [[] for _ in range(max_iters)]
    solved_iter = None
    ever_unit_ok = False
    #cur_pass = 0
    is_first_reflection = True
    is_solved = False
    reflections, implementations, test_feedback = [], [], []
    cur_func_impl = item["bug_source_code"]
    cur_feedback = None

    reflections.append("")
    func_impls = []
    batch_size = 1
    for b in range(0, n_completions, batch_size):
        impls = generate_function(
        gen,
        item,
        model,
        strategy="few_shot",
        cur_func_impl=item["bug_source_code"],
        problem_context=create_problem_template(item, False),
        # inferred_specificaion=inferred_specificaion,
        reflections=reflections,
        is_first_reflection=is_first_reflection,
        prompting=prompting,
        feedback=None,
        num_comps=batch_size   
    )
        print("File few-shot: ",f"impls: {impls}")
        
        func_impls.extend(impls)
    
    is_first_reflection = False

    for cur_impl in func_impls:
        implementations.append(cur_impl)
        cur_func_impl = cur_impl

        if isinstance(item["hidden_unit_tests"], str):
            item["hidden_unit_tests"] = json.loads(item["hidden_unit_tests"])

        hidden_tests = item["hidden_unit_tests"]
        print(f"File few-shot hidden_tests: {hidden_tests}")

        unit_ok, unit_test_results = exe.evaluate(cur_func_impl, item["hidden_unit_tests"], timeout=10)
        ever_unit_ok = ever_unit_ok or unit_ok
        print("File few-shot: ",f"unit_test_results first: {unit_test_results}")
        print("File few-shot: ",f"unit_ok first: {unit_ok}")
        # test_feedback.append(f"unit_tests_passed={unit_ok}")
        iteration_unit_pass_matrix[0].append(unit_ok)
        if unit_ok:
            solved_iter = 0
            is_solved = True
            # break
        else:
            is_solved = False

    #     cur_iter = 1

    #     while not is_solved and cur_iter < max_iters:
    #         print("File few-shot: ",f"cur_iter: {cur_iter}")
    #         next_impls = generate_function(
    #             gen,
    #             item,
    #             model,
    #             strategy="few_shot",
    #             cur_func_impl=cur_func_impl,
    #             problem_context=create_problem_template(item, False),
    #             # inferred_specificaion=inferred_specificaion,
    #             reflections=reflections,
    #             is_first_reflection=is_first_reflection,
    #             prompting=prompting,
    #             feedback=cur_feedback,
    #             num_comps=1
    #         )
    #         cur_func_impl = next_impls[0]
    #         implementations.append(cur_func_impl)

    #         if isinstance(item["hidden_unit_tests"], str):
    #             item["hidden_unit_tests"] = json.loads(item["hidden_unit_tests"])

    #         unit_ok, unit_test_results = exe.evaluate(cur_func_impl, item["hidden_unit_tests"], timeout=10)
    #         print("File few-shot: ",f"unit_ok 2: {unit_ok}")
    #         print("File few-shot: ",f"unit_test_results 2: {unit_test_results}")
    #         ever_unit_ok = ever_unit_ok or unit_ok
    #         iteration_unit_pass_matrix[cur_iter].append(unit_ok)

    #         if cur_iter == max_iters - 1:
    #             if unit_ok:
    #                 solved_iter = cur_iter
    #                 is_solved = True
    #             break

    #         cur_iter += 1

    # if ever_unit_ok:
    #     success_count += 1
    #     is_solved = True

    item["is_solved"] = is_solved
    item["implementations"] = implementations
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
    
    # for iter_idx, results in enumerate(iteration_pass_matrix):
    #     c = sum(results)
    #     item[f"pass@{pass_at_k}_iter{iter_idx}"] = codex_pass_at_k(n_completions, c, pass_at_k)
    #     print("File refl_omission: ",f"pass@{pass_at_k}_iter{iter_idx}: {codex_pass_at_k(n_completions, c, pass_at_k)}")

    for iter_idx, results in enumerate(iteration_unit_pass_matrix):
        c = sum(results)
        item[f"pass@{pass_at_k}_unit_iter{iter_idx}"] = codex_pass_at_k(n_completions, c, pass_at_k)
        print("File refl_omission: ",f"pass@{pass_at_k}_unit_iter{iter_idx}: {codex_pass_at_k(n_completions, c, pass_at_k)}")

    return item, item[f"pass@{pass_at_k}"]

def run_few_shot(
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