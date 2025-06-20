# generator_utils.py (clean and separated version)

from generators.model import ModelBase, Message
from typing import Union, List, Optional, Callable, Tuple
import json
import random
import re

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

def sample_n_random(items: List[str], n: int) -> List[str]:
    if n >= len(items):
        return items
    return random.sample(items, n)

def extract_json(raw_output):
    start_index = raw_output.find("[")
    if start_index == -1:
        raise ValueError("No valid JSON array found!")
    return json.loads(raw_output[start_index:])

def extract_json_fuzzy(output: str):
    try:
        # Try strict parsing first
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Try to extract the JSON array manually using regex
    match = re.search(r"\[\s*{.*?}\s*]", output, re.DOTALL)
    if match:
        json_block = match.group(0)
        try:
            return json.loads(json_block)
        except json.JSONDecodeError:
            raise ValueError("Found potential JSON block but failed to decode.")
    
    raise ValueError("No valid JSON array found in model output.")


def generic_generate_func_impl(
    problem_context: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    reflections,
    is_first_reflection: bool,
    inferred_specificaion:str,
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
    reflexion_chat_instruction_test_omit: str,
    reflexion_few_shot_test_omit: str,
    first_reflexion_chat_instruction_first_omit: str,
    first_reflexion_few_shot_first_omit: str,
    reflexion_chat_instruction_self_omit: str,
    reflexion_few_shot_self_omit: str,

    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str],
) -> Union[str, List[str]]:

    if model.is_chat:
        if strategy == "reflexion":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[reflection on previous impl]:\n{reflections}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=message),
                    Message(role="assistant", content=reflections),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code based on the reflection?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
            else:
                prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}\n\n{reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="assistant", content=f"[previous impl]:\n{add_code_block(prev_func_impl)}"),
                    Message(role="user", content=f"[problem description]:\n{problem_context}"),
                    Message(role="assistant", content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{reflections}"),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code based on the reflection?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
        elif strategy == "simple":
            prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(prompt, "[improved impl]:")
            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content=f"[problem description]:\n{problem_context}"),
                Message(role="user", content="[improved impl]:")
            ]
        elif strategy == "test_gen_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[reflection on previous impl]:\n{reflections}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=message),
                    Message(role="assistant", content=reflections),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code based on the reflection?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
            else:
                prompt = f"{reflexion_chat_instruction_test_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_test_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="assistant", content=f"[previous impl]:\n{add_code_block(prev_func_impl)}"),
                    Message(role="user", content=f"[problem description]:\n{problem_context}"),
                    Message(role="assistant", content=f"[reflection on previous impl]:\n{reflections}"),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code based on the reflection?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
        elif strategy == "first_refl_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction_first_omit}\n{code_block_instruction}\n\n{first_reflexion_few_shot_first_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=message),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
            else:
                prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}\n\n{reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="assistant", content=f"[previous impl]:\n{add_code_block(prev_func_impl)}"),
                    Message(role="user", content=f"[problem description]:\n{problem_context}"),
                    Message(role="assistant", content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{reflections}"),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code based on the reflection?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
        elif strategy == "self_refl_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[reflection on previous impl]:\n{reflections}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=message),
                    Message(role="assistant", content=reflections),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code based on the reflection?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
            else:
                prompt = f"{reflexion_chat_instruction_self_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_self_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="assistant", content=f"[previous impl]:\n{add_code_block(prev_func_impl)}"),
                    Message(role="user", content=f"[problem description]:\n{problem_context}"),
                    Message(role="assistant", content=f"[unit test results from previous impl]:\n{feedback}"),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
        elif strategy == "refl_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction_first_omit}\n{code_block_instruction}\n\n{first_reflexion_few_shot_first_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=message),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
            else:
                prompt = f"{reflexion_chat_instruction_self_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_self_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="assistant", content=f"[previous impl]:\n{add_code_block(prev_func_impl)}"),
                    Message(role="user", content=f"[problem description]:\n{problem_context}"),
                    Message(role="assistant", content=f"[unit test results from previous impl]:\n{feedback}"),
                    Message(role="user", content=(
                        "Before writing the improved implementation, please answer:\n"
                        "1. What specific changes are you going to make to the code?\n"
                        "2. How exactly will this change address the issue?\n\n"
                        "Then, write your full improved implementation in Ruby.\n"
                        "Ensure that your code actually reflects the reasoning above and addresses the problem."
                    ))
                ]
        func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{reflections}\n\n# improved implementation\n{code_block_instruction}"
        else:
            prompt = f"{simple_completion_instruction}\n{code_block_instruction}"
        func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        parsed = parse_code_block(func_bodies)
        print_generated_func_body(parsed)
        return parsed
    else:
        parsed = [parse_code_block(body) for body in func_bodies]
        print_generated_func_body("\n\n".join(parsed))
        return parsed

def generic_generate_scot_func_impl(
    *args, **kwargs
):
    return generic_generate_func_impl(*args, **kwargs)

def generic_generate_internal_tests(
    samples: List[str],
    problem_context: str,
    model: ModelBase,
    max_num_tests: int,
    test_generation_few_shot: str,
    test_generation_chat_instruction: str,
    test_generation_completion_instruction: str,
    inferred_specificaion: str,
    num_comps: int,
    is_react: bool = False,
) -> Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]]:
    if model.is_chat:
        if is_react:
            messages = [
                Message(role="system", content=test_generation_chat_instruction),
                Message(role="user", content=f"{test_generation_few_shot}\n\n[think]:")
            ]
        else:
            prompt = f"{test_generation_chat_instruction}\n{test_generation_few_shot}"
            message = f"[problem context]:\n{problem_context}\n\n[test case samples]:\n{samples}\n\n[unit tests]:"
            print_messages(prompt, message)
            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content=message)
            ]
        outputs = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature, max_tokens=1024)
        if isinstance(outputs, str):
            outputs = [outputs]
    else:
        prompt = f"{test_generation_completion_instruction}\nunit tests:"
        outputs = model.generate(prompt, num_comps=num_comps, temperature=temperature, max_tokens=1024)
        if isinstance(outputs, str):
            outputs = [outputs]

    all_tests = []
    for output in outputs:
        try:
            unit_tests = extract_json_fuzzy(output)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Test generation failed: {e}")
            all_tests.append([])
            continue

        if isinstance(unit_tests, list):
            extracted = sample_n_random([
                (t["input"], t["output"])
                for t in unit_tests if isinstance(t, dict) and "input" in t and "output" in t
            ], max_num_tests)
            all_tests.append(extracted)
        else:
            all_tests.append([])

    return all_tests[0] if num_comps == 1 else all_tests


def extract_validated_tests_from_cot_response(
    response: Union[str, List[str]]
) -> List[Tuple[str, str]]:
    """
    Extracts test cases marked as correct (✅) from CoT-style LLM output.
    Supports both single response (str) and batched responses (List[str]).
    Returns a list of (input, output) tuples.
    """
    pattern = re.compile(
        r"\*\*Input:\*\*\s*`(.*?)`\s*\*\*Expected Output:\*\*\s*`(.*?)`.*?\*\*Verdict:\*\*\s*✅ Correct output",
        re.DOTALL
    )

    if isinstance(response, str):
        response = [response]

    validated_tests = []
    for r in response:
        validated_tests.extend((inp.strip(), out.strip()) for inp, out in pattern.findall(r))

    return validated_tests


def generic_validate_internal_tests(
    tests: List[Tuple[str, List[str]]],
    problem_context: str,
    func: str,
    model: ModelBase,
    max_num_tests: int,
    test_generation_few_shot: str,
    test_generation_chat_instruction: str,
    test_generation_completion_instruction: str,
    inferred_specificaion: str,
    is_react: bool = False,
    num_comps: int = 1  # <--- Added here
) -> List[Tuple[str, str]]:
    """
    Validates internal test cases using the provided model.
    Returns a list of (input, expected_output) tuples that are marked as correct.
    """
    formatted_tests = [
        {"input": input_str, "output": "\n".join(output_list)}
        for input_str, output_list in tests
    ]

    if model.is_chat:
        prompt = f"{test_generation_chat_instruction}\n{test_generation_few_shot}"
        message = f"""[buggy code]:
{func}

[problem context]:
{problem_context}

[Test cases to validate]:
{json.dumps(formatted_tests, indent=2)}

[unit tests]:"""

        print_messages(prompt, message)
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=message)
        ]

        output = model.generate_chat(messages=messages, max_tokens=1024)
        print(f"OUTPUT VALIDATION!!!!!!: {output}")

        # If batched, pass output as list; else pass as string
        validated = extract_validated_tests_from_cot_response(
            [output] if num_comps > 1 else output
        )

        unit_tests = [{"input": inp, "output": out} for inp, out in validated]
    else:
        # Handle non-chat model (optional or stub)
        prompt = f"{test_generation_completion_instruction}\n\nfunc signature:\n{func}\nunit tests:"
        output = model.generate(prompt, max_tokens=1024)
        unit_tests = []

    if isinstance(unit_tests, list):
        return sample_n_random([
            (t["input"], t["output"])
            for t in unit_tests if isinstance(t, dict) and "input" in t and "output" in t
        ], max_num_tests)

    return []

def generic_generate_self_reflection(
    func: str,
    feedback: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    self_reflection_chat_instruction_test_omit: str,
    self_reflection_few_shot_test_omit: str,
    inferred_specificaion: str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str],
    strategy: str,
    num_comps: int = 1
) -> Union[str, List[str]]:
    is_reflexion = strategy in {"reflexion", "first_refl_omission", "self_refl_omission"}
    
    if model.is_chat:
        if is_reflexion:
            system_content = self_reflection_chat_instruction
            if self_reflection_few_shot:
                system_content += f"\n{self_reflection_few_shot}"
            user_content = f"[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:"
        elif strategy == "test_gen_omission":
            system_content = self_reflection_chat_instruction_test_omit
            if self_reflection_few_shot_test_omit:
                system_content += f"\n{self_reflection_few_shot_test_omit}"
            user_content = f"[function impl]:\n{add_code_block(func)}\n\n[self-reflection]:"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print_messages(system_content, user_content)

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]

        if num_comps > 1:
            return [model.generate_chat(messages=messages) for _ in range(num_comps)]
        else:
            return model.generate_chat(messages=messages)

    # For non-chat models (completion-based)
    prompt = f"{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:"
    if num_comps > 1:
        return [model.generate(prompt) for _ in range(num_comps)]
    return model.generate(prompt)

def generic_generate_first_reflection(
    problem_context: str,
    func: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    inferred_specificaion:str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str],
) -> str:
    if model.is_chat:
        system_content = self_reflection_chat_instruction
        if self_reflection_few_shot:
            system_content += f"\n{self_reflection_few_shot}"
        user_content = f"[incorrect function impl]:\n{add_code_block(func)}\n\n[problem context]:\n{problem_context}\n\n[self-reflection]:"
        print_messages(system_content, user_content)
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]
        response = model.generate_chat(messages=messages)
        print(f"RESPONSE!!!!!!: {response}")
        return response
    return model.generate(f"{self_reflection_completion_instruction}\n{func}\n\nExplanation:")

def generic_generate_first_reflection(
    problem_context: str,
    func: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    inferred_specificaion: str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str],
    num_comps: int = 1  # <--- Added
) -> Union[str, List[str]]:
    if model.is_chat:
        system_content = self_reflection_chat_instruction
        if self_reflection_few_shot:
            system_content += f"\n{self_reflection_few_shot}"
        user_content = (
            f"[incorrect function impl]:\n{add_code_block(func)}\n\n"
            f"[problem context]:\n{problem_context}\n\n[self-reflection]:"
        )

        print_messages(system_content, user_content)

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]

        if num_comps > 1:
            responses = [model.generate_chat(messages=messages) for _ in range(num_comps)]
            for i, r in enumerate(responses):
                print(f"RESPONSE {i + 1}!!!!!!: {r}")
            return responses
        else:
            response = model.generate_chat(messages=messages)
            print(f"RESPONSE!!!!!!: {response}")
            return response

    # Non-chat model (completion-based)
    prompt = f"{self_reflection_completion_instruction}\n{func}\n\nExplanation:"
    if num_comps > 1:
        return [model.generate(prompt) for _ in range(num_comps)]
    return model.generate(prompt)
