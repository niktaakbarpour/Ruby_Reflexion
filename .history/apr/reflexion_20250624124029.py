from typing import List
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

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
        return gen.scot_func_impl(
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
        return gen.func_impl(
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
        verbose,
        # infer_spec
    ):

    print_v = make_printv(verbose)
    is_first_reflection = True
    success_count = 0
    iteration_pass_matrix = [[] for _ in range(max_iters)]

    is_solved = False
    reflections, implementations, test_feedback = [], [], []
    cur_func_impl = item["bug_source_code"]
    is_first_reflection = True
    cur_feedback = None

    # if infer_spec:
    #     inferred_specificaion = gen.infer_specification(
    #         problem_context=create_problem_template(item, False),
    #         model=model,
    #     )
                
    reflection = gen.first_reflection(
        problem_context=create_problem_template(item, False),
        # inferred_specificaion=inferred_specificaion,
        func=item["bug_source_code"],
        model=model
    )
    reflections.append(reflection)

    samples = [(inp.replace(" ", "\n") + '\n', out)
        for inp, out in zip(item["sample_inputs"], item["sample_outputs"])]

    tests = gen.internal_tests(
        problem_context=create_problem_template(item, False),
        # inferred_specificaion=inferred_specificaion,
        model=model,
        max_num_tests=7,
        samples=samples,
    )
    print(f"tests_i: {tests}")
    # formatted_tests = [{"input": inp, "output": out} for inp, out in tests]
    formatted_tests = [
    {
        "input": ' '.join(inp.strip().split()) + '\n',
        "output": ''.join(out).strip() if isinstance(out, list) else out.strip()
    }
    for inp, out in tests
]

    print(f"formatted_tests: {formatted_tests}")

    # validated_tests = gen.validate_internal_tests(
    #     tests=tests,
    #     problem_context=create_problem_template(item, False),
    #     func=item["bug_source_code"],
    #     model=model,
    #     max_num_tests=5
    # )
    # print(f"validated_tests_i: {validated_tests}")

    func_impls = []
    batch_size = 1
    for b in range(0, n_completions, batch_size):
        impls = generate_function(
        gen,
        item,
        model,
        strategy="reflexion",
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
        print(f"cur_impl: {cur_impl}")
        result = exe.execute(cur_impl, formatted_tests)
        is_passing = result["is_passing"]
        print(f"is_passing: {is_passing}")
        feedback = result["feedback"]
        print(f"feedback: {feedback}")
        test_feedback.append(feedback)
        iteration_pass_matrix[0].append(is_passing)
        if is_passing and exe.evaluate(cur_impl, item["unittest_cases"], timeout=10):
            success_count += 1
            is_solved = True
            continue

        cur_func_impl = cur_impl
        cur_iter = 1
        cur_feedback = feedback

        while not is_solved and cur_iter < max_iters:
            print("IN LOOP")
            reflection = gen.self_reflection(
                problem_context=create_problem_template(item, False),
                # inferred_specificaion=inferred_specificaion,
                func=cur_func_impl,
                feedback=cur_feedback,
                model=model,
                strategy="reflexion"
            )
            reflections.append(reflection)
            print(f"REFLECTION!!!!!!!!: {reflection}")
            next_impls = generate_function(
                gen,
                item,
                model,
                strategy="reflexion",
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

    item["is_solved"] = is_solved
    item["reflections"] = reflections
    item["implementations"] = implementations
    item["test_feedback"] = test_feedback
    item["solution"] = cur_func_impl
    item["success_count"] = success_count
    item[f"pass@{pass_at_k}"] = codex_pass_at_k(n_completions, success_count, pass_at_k)

    for iter_idx, results in enumerate(iteration_pass_matrix):
        c = sum(results)
        item[f"pass@{pass_at_k}_iter{iter_idx}"] = codex_pass_at_k(n_completions, c, pass_at_k)

    return item, item[f"pass@{pass_at_k}"]



def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    # infer_spec: bool,
    is_leetcode: bool = False,
    model_path: str = None,
) -> None:
    prompting = "cot"
    # infer_spec = False
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name, model_path)

    print_v = make_printv(verbose)

    num_items = len(dataset)

    total_success = resume_success_count(dataset)
    n_completions = 1 #n
    k = pass_at_k
    pass10_list = []

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
            pass10_list.append(updated_item[f"pass@{pass_at_k}"])
            print_v(f"completed {i+1}/{num_items}: pass@10 so far = {round(sum(pass10_list)/(i+1), 3)}")
        except Exception as e:
            print(f"Error processing item {i}: {e}. Continuing.")
            continue

    if pass10_list:
        overall_pass10 = sum(pass10_list) / len(pass10_list)
        print(f"\n🟢 FINAL pass@{pass_at_k} across all {len(pass10_list)} bugs: {overall_pass10:.3f}")
    else:
        print(f"\n⚠️  No bugs were processed, so pass@{pass_at_k} cannot be computed.")
