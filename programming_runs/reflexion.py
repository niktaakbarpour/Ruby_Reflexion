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
    print("I am here 1.")
    exe = executor_factory(language, is_leet=is_leetcode)
    print("I am here 1.1")
    gen = generator_factory(language)
    print("I am here 1.2")
    model = model_factory(model_name, model_path)
    print("I am here 1.3")

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        print("I am here 2.")
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        while cur_pass < pass_at_k and not is_solved:
            print("I am here 3.")
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                print("I am here 4.")
                print(f"item_prompt: {item['prompt']}")
                tests_i = gen.internal_tests(item["prompt"], model, 1)

            print("I'm here.5")

            # first attempt
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)


            print(f"passing: {is_passing}")
            # if solved, exit early
            if is_passing:
                print("I'm here.6")
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                num_success += int(is_passing)
                break

            # use self-reflection to iteratively improve
            cur_iter = 1
            cur_feedback = feedback
            print("I'm here.7")
            while cur_iter < max_iters:
                # get self-reflection
                reflection = gen.self_reflection(
                    cur_func_impl, cur_feedback, model)
                reflections += [reflection]

                print("I'm here.8")

                # apply self-reflection in the next attempt
                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                print(f"Type of cur_func_impl: {type(cur_func_impl)}")
                print(f"Value of cur_func_impl: {cur_func_impl}")
                assert isinstance(cur_func_impl, str)

                # check if all internal unit tests pass
                is_passing, cur_feedback, _ = exe.execute(
                    cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                # if solved, check if it passes the real tests, exit early
                if is_passing or cur_iter == max_iters - 1:
                    print("I'm here.9")
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        print("I'm here.10")
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
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
