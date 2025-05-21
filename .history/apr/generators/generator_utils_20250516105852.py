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
    if strategy not in {"reflexion", "simple"}:
        raise ValueError(f"Invalid strategy: {strategy}")

    if model.is_chat:
        if strategy == "reflexion":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[reflection on previous impl]:\n{self_reflection}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=message),
                    Message(role="assistant", content=self_reflection),
                    Message(role="user", content="[improved impl]:")
                ]
            else:
                prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}\n\n{reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                print_messages(prompt, message)
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="assistant", content=add_code_block(prev_func_impl)),
                    Message(role="user", content=problem_context),
                    Message(role="assistant", content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}"),
                    Message(role="user", content="[improved impl]:")
                ]
        else:
            prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(prompt, "[improved impl]:")
            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content="[improved impl]:")
            ]

        func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{code_block_instruction}"
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
    func: str,
    model: ModelBase,
    max_num_tests: int,
    test_generation_few_shot: str,
    test_generation_chat_instruction: str,
    test_generation_completion_instruction: str,
    is_react: bool = False,
) -> List[str]:
    if model.is_chat:
        if is_react:
            messages = [
                Message(role="system", content=test_generation_chat_instruction),
                Message(role="user", content=f"{test_generation_few_shot}\n\n[func signature]:\n{func}\n\n[think]:")
            ]
        else:
            prompt = f"{test_generation_chat_instruction}\n{test_generation_few_shot}"
            message = f"[buggy code]:\n{func}\n\n[problem context]:\n{problem_context}\n\n[test case samples]:\n{samples}\n\n[unit tests]:"
            print_messages(prompt, message)
            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content=message)
            ]
        output = model.generate_chat(messages=messages, max_tokens=1024)
        print(f"OUTPUT GENERATION!!!!!!: {output}")
        try:
            unit_tests = extract_json_fuzzy(output)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Test generation failed: {e}")
            return []
    else:
        prompt = f"{test_generation_completion_instruction}\n\nfunc signature:\n{func}\nunit tests:"
        output = model.generate(prompt, max_tokens=1024)
        unit_tests = []

    if isinstance(unit_tests, list):
        return sample_n_random([
            (t["input"], t["output"])
            for t in unit_tests if isinstance(t, dict) and "input" in t and "output" in t
        ], max_num_tests)
    return []

def extract_validated_tests_from_cot_response(response: str) -> List[Tuple[str, str]]:
    """
    Extracts test cases marked as correct (✅) from CoT-style LLM output.
    Returns a list of (input, output) tuples.
    """
    pattern = re.compile(
        r"\*\*Input:\*\*\s*`(.*?)`\s*\*\*Expected Output:\*\*\s*`(.*?)`.*?\*\*Verdict:\*\*\s*✅ Correct output",
        re.DOTALL
    )
    return [(inp.strip(), out.strip()) for inp, out in pattern.findall(response)]


def generic_validate_internal_tests(
    tests: List[Tuple[str, List[str]]],
    problem_context: str,
    func: str,
    model: ModelBase,
    max_num_tests: int,
    test_generation_few_shot: str,
    test_generation_chat_instruction: str,
    test_generation_completion_instruction: str,
    is_react: bool = False,
) -> List[Tuple[str, str]]:
    formatted_tests = []
    for input_str, output_list in tests:
        # Assuming output is always a list of strings
        expected_output = "\n".join(output_list)
        formatted_tests.append({
            "input": input_str,
            "output": expected_output
        })

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
        validated = extract_validated_tests_from_cot_response(output)
        unit_tests = [{"input": inp, "output": out} for inp, out in validated]
    else:
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
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str],
) -> str:
    if model.is_chat:
        system_content = self_reflection_chat_instruction
        if self_reflection_few_shot:
            system_content += f"\n{self_reflection_few_shot}"
        user_content = f"[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:"
        print_messages(system_content, user_content)
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]
        return model.generate_chat(messages=messages)
    return model.generate(f"{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:")

def generic_generate_first_reflection(
    problem_context: str,
    func: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
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

