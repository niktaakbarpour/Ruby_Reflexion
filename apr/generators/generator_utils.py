from generators.model import ModelBase, Message
import random

from typing import Union, List, Optional, Callable
import json

def generic_generate_func_impl(
    problem_context: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    self_reflection,
    is_first_reflection: bool,
    feedback,
    num_comps,
    temperature,
    reflexion_chat_instruction: str,
    first_reflexion_chat_instruction: str,
    reflexion_few_shot: str,
    first_reflexion_few_shot: str,
    simple_chat_instruction: str,
    reflexion_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str],
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    # if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
    #     raise ValueError(
    #         f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model.is_chat:
        if strategy == "reflexion":
            if is_first_reflection == True:
                print(f"is_first_reflection: {is_first_reflection}")
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:"
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                print_messages(prompt, message)
                # func_bodies is a really bad name, as it can also be just 1 string
                messages = [
                    Message(
                        role="system",
                        content=prompt,
                    ),
                    Message(
                        role="user", # TODO: check this
                        content=f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}",
                    ),
                    Message(
                        role="assistant",
                        content=self_reflection,
                    ),
                    Message(
                        role="user",
                        content=f"[improved impl]:",
                    ),
                ]
                func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
            else:
                print(f"is_first_reflection2: {is_first_reflection}")
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:"
                prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}\n\n{reflexion_few_shot}"
                print_messages(prompt, message)
                # func_bodies is a really bad name, as it can also be just 1 string
                messages = [
                    Message(
                        role="system",
                        content=prompt,
                    ),
                    Message(
                        role="assistant",
                        content=add_code_block(prev_func_impl),
                    ),
                    Message(
                        role="user",
                        content=problem_context,
                    ),
                    Message(
                        role="assistant",
                        content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}",
                    ),
                    Message(
                        role="user",
                        content=f"[improved impl]:",
                    ),
                ]
                func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        else:
            system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(system_prompt, "[improved impl]:")
            messages = [
                Message(
                    role="system",
                    content=f"{simple_chat_instruction}\n{code_block_instruction}",
                ),
                Message(
                    role="user",
                    content="[improved impl]:",
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = f"{simple_completion_instruction}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies


def generic_generate_internal_tests(
        problem_context: str,
        func: str,
        model: ModelBase,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        test_generation_completion_instruction: str,
        # parse_tests: Callable[[str], List[str]],
        # is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """Generates tests for a function."""
    if model.is_chat:
        if is_react:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func}\n\n[think]:"
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
        else:
            message = f"[buggy code]:\n{func}\n\n[problem context]:\n{problem_context}\n\n[unit tests]:"
            prompt = f"{test_generation_chat_instruction}\n{test_generation_few_shot}"
            print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=f"{test_generation_chat_instruction}\n{test_generation_few_shot}",
                ),
                Message(
                    role="user",
                    content=f"[buggy code]:\n{func}\n\n[problem context]:\n{problem_context}\n\n[unit tests]:"
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
    else:
        prompt = f'{test_generation_completion_instruction}\n\nfunc signature:\n{func}\nunit tests:'
        output = model.generate(prompt, max_tokens=1024)
    # all_tests = parse_tests(output.split("\n"))
    # valid_tests = [test for test in all_tests if is_syntax_valid(test)]
    
    # Remove triple backticks and surrounding whitespace
    cleaned_output = output.strip("`").strip()

    # If the first line is "json", remove it
    lines = cleaned_output.split("\n")
    if lines[0].strip().lower() == "json":
        cleaned_output = "\n".join(lines[1:])  # Remove first line

    # Handle empty output case
    if not cleaned_output:
        print("Warning: Model output is empty.")
        return []

    # Parse JSON safely
    try:
        parsed_output = json.loads(cleaned_output)  # Convert JSON string to Python object
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []  # Return an empty list if parsing fails

    # Extract raw test cases (inputs and outputs)
    filtered_tests = []
    for test_case in parsed_output:
        if "input" in test_case and "output" in test_case:
            filtered_tests.append((test_case["input"], test_case["output"]))

    return sample_n_random(filtered_tests, max_num_tests)


def generic_generate_self_reflection(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
) -> str:
    if model.is_chat:
        if self_reflection_few_shot is not None:
            prompt = f"{self_reflection_chat_instruction}\n{self_reflection_few_shot}"
            message = f"[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:"
            print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=f"{self_reflection_chat_instruction}\n{self_reflection_few_shot}",
                ),
                Message(
                    role="assistant",
                    content=f"[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}",
                ),
                Message(
                    role="user",
                    content=f"[self-reflection]:",
                ),
            ]
            reflection = model.generate_chat(messages=messages)
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    else:
        reflection = model.generate(
            f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:')
    return reflection  # type: ignore

def generic_generate_first_reflection(
        problem_context: str,
        func: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
) -> str:
    if model.is_chat:
        if self_reflection_few_shot is not None:
            prompt = f"{self_reflection_chat_instruction}\n\n{self_reflection_few_shot}"
            message = f"[incorrect function impl]:\n{add_code_block(func)}\n\n[problem context]:\n{problem_context}\n\n[self-reflection]:"
            print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=f"{self_reflection_chat_instruction}\n\n{self_reflection_few_shot}",
                ),
                Message(
                    role="user",
                    content=f"[incorrect function impl]:\n{add_code_block(func)}\n\n[problem context]:\n{problem_context}\n\n[self-reflection]:",
                )
            ]
            reflection = model.generate_chat(messages=messages)
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{func}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    else:
        reflection = model.generate(
            f'{self_reflection_completion_instruction}\n{func}\n\nExplanation:')
    return reflection  # type: ignore


def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)

def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)

def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")
