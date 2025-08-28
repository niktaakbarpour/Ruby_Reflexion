# generator_utils.py (clean and separated version)

from generators.model import ModelBase, Message
from typing import Union, List, Optional, Callable, Tuple
import json
import random
import re
from .prompt_constants import (
    FIRST_REFLECTION_INFER_VS_SUMMARY_CHAT_INSTRUCTION,
    FIRST_REFLECTION_INFER_VS_SUMMARY_FEW_SHOT
)

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


import json
import re
from typing import List, Dict

def _slice_json_array_payload(text: str) -> str:
    # Prefer a fenced block
    m = re.search(r"```json\s*(.*?)```", text, flags=re.S | re.I)
    payload = m.group(1) if m else text

    # Find the first '[' (start of array)
    start = payload.find('[')
    if start == -1:
        raise ValueError("No '[' found for JSON array.")
    return payload[start:]  # from '[' onward

def extract_first_n_tests(text: str, n: int) -> List[Dict[str, str]]:
    """
    Extract up to N objects from the first JSON array present in text.
    Robust to extra items and truncated tails.
    Returns exactly up to N items; if fewer than N are found, returns what it found.
    """
    arr_text = _slice_json_array_payload(text)

    # Walk the text to collect balanced {...} objects at top array level.
    i = 0
    in_string = False
    escape = False
    depth = 0
    objs: List[str] = []
    current_obj_start = None

    # Skip the initial '['
    if not arr_text or arr_text[0] != '[':
        raise ValueError("JSON array does not start with '['.")
    i = 1

    while i < len(arr_text) and len(objs) < n:
        ch = arr_text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        # not in string
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == '{':
            if depth == 0:
                current_obj_start = i
            depth += 1
            i += 1
            continue

        if ch == '}':
            depth -= 1
            if depth == 0 and current_obj_start is not None:
                # We have a complete object substring
                obj_str = arr_text[current_obj_start:i+1]
                objs.append(obj_str)
                current_obj_start = None
            i += 1
            continue

        # End of array?
        if ch == ']' and depth == 0:
            break

        # Otherwise move on (commas, spaces, etc.)
        i += 1

    # Now parse objects we collected
    results: List[Dict[str, str]] = []
    for obj_str in objs[:n]:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "input" in obj and "output" in obj:
                results.append({
                    "input": str(obj["input"]),
                    "output": str(obj["output"])
                })
        except json.JSONDecodeError:
            # Ignore malformed objects (can happen if we somehow sliced poorly)
            continue

    return results[:n]



def generic_generate_func_impl(
    problem_context: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    reflections,
    is_first_reflection: bool,
    # inferred_specificaion:str,
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
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
            else:
                prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}\n\n{reflexion_few_shot}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[unit test results from previous impl]:\n{feedback}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
                print_messages(prompt, user_content)
        elif strategy == "simple":
            prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            
            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content=f"[problem description]:\n{problem_context}"),
                Message(role="user", content="[improved impl]:")
            ]
        elif strategy == "test_gen_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
            else:
                prompt = f"{reflexion_chat_instruction_test_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_test_omit}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
        elif strategy == "first_refl_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction_first_omit}\n{code_block_instruction}\n\n{first_reflexion_few_shot_first_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                print(f"prompt1: {prompt}")

                print(f"user_content1: {user_content}")

                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
            else:
                prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}\n\n{reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[unit test results from previous impl]:\n{feedback}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                print(f"prompt2: {prompt}")

                print(f"user_content2: {user_content}")

                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]

        elif strategy == "self_refl_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction}\n{code_block_instruction}\n\n{first_reflexion_few_shot}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}\n\n[reflection on previous impl]:\n{reflections}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
            else:
                prompt = f"{reflexion_chat_instruction_self_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_self_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[unit test results from previous impl]:\n{feedback}",
                    f"[hint for changing the implementation]:\n{reflections}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]

        elif strategy == "refl_omission":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction_first_omit}\n{code_block_instruction}\n\n{first_reflexion_few_shot_first_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
                print(f"generator messages: {messages}")
            else:
                prompt = f"{reflexion_chat_instruction_self_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_self_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[unit test results from previous impl]:\n{feedback}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]

        elif strategy == "few_shot":
            if is_first_reflection:
                prompt = f"{first_reflexion_chat_instruction_first_omit}\n{code_block_instruction}\n\n{first_reflexion_few_shot_first_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
                print(f"generator messages: {messages}")
            else:
                prompt = f"{reflexion_chat_instruction_self_omit}\n{code_block_instruction}\n\n{reflexion_few_shot_self_omit}"
                message = f"[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[problem context]:\n{problem_context}"
                user_content = "\n\n".join([
                    f"[previous implementation]:\n{add_code_block(prev_func_impl)}",
                    f"[problem description]:\n{problem_context}",
                    f"[unit test results from previous impl]:\n{feedback}",
                    "Write your full improved implementation in Ruby:\n"
                ])
                messages = [
                    Message(role="system", content=prompt),
                    Message(role="user", content=user_content),
                ]
        func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        print(f"generator func_bodies: {func_bodies}")

    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{reflections}\n\n# improved implementation\n{code_block_instruction}"
        else:
            prompt = f"{simple_completion_instruction}\n{code_block_instruction}"
        func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        # normalize to string
        if isinstance(func_bodies, list):
            func_bodies = func_bodies[0] if func_bodies else ""

        assert isinstance(func_bodies, str)

        for idx, ln in enumerate(str(func_bodies).splitlines()):
            if "```" in ln or "~~~" in ln:
                print(idx, repr(ln))


        parsed = parse_code_block(func_bodies)
        print("parsed:", parsed)
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
    # inferred_specificaion: str,
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
        outputs = model.generate_chat(messages=messages, max_tokens=1024)
        print(f"outputs: {outputs}")
        if isinstance(outputs, str):
            outputs = [outputs]
            print(f"outputs2: {outputs}")
    else:
        prompt = f"{test_generation_completion_instruction}\nunit tests:"
        outputs = model.generate(prompt, max_tokens=1024, temperature=0.1)
        if isinstance(outputs, str):
            outputs = [outputs]

    all_tests = []
    for output in outputs:
        # extract first N tests even if model kept talking or truncated the tail
        tests = extract_first_n_tests(output, max_num_tests)
        print(f"extracted_tests: {tests}")
        # If you still want random sampling within those, do it here;
        # but typically we want exactly N, so just take them as-is:
        pairs = [(t["input"], t["output"]) for t in tests]
        all_tests.append(pairs)

    print(f"all_tests: {all_tests}")
    return all_tests[0]


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
    # inferred_specificaion: str,
    is_react: bool = False,
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
            output
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
    problem_context: str,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    self_reflection_chat_instruction_test_omit: str,
    self_reflection_few_shot_test_omit: str,
    # inferred_specificaion: str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str],
    strategy: str,
) -> Union[str, List[str]]:
    is_reflexion = strategy in {"reflexion", "first_refl_omission", "self_refl_omission"}
    
    if model.is_chat:
        if is_reflexion:
            system_content = self_reflection_chat_instruction
            if self_reflection_few_shot:
                system_content += f"\n{self_reflection_few_shot}"
            user_content = f"[problem context]:\n{problem_context}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:"
        elif strategy == "test_gen_omission":
            system_content = self_reflection_chat_instruction_test_omit
            if self_reflection_few_shot_test_omit:
                system_content += f"\n{self_reflection_few_shot_test_omit}"
            user_content = f"[problem context]:\n{problem_context}\n\n[function impl]:\n{add_code_block(func)}\n\n[self-reflection]:"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print_messages(system_content, user_content)

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]
        return model.generate_chat(messages=messages)

    # For non-chat models (completion-based)
    prompt = f"{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:"
    return model.generate(prompt)

def generic_generate_first_reflection(
    problem_context: str,
    func: str,
    model: ModelBase,
    self_reflection_chat_instruction: str,
    self_reflection_completion_instruction: str,
    add_code_block: Callable[[str], str],
    self_reflection_few_shot: Optional[str],
    inferred_specificaion: Optional[str] = None,
    code_summary: Optional[str] = None,
) -> Union[str, List[str]]:

    if model.is_chat:
        if inferred_specificaion and code_summary:
            self_reflection_chat_instruction = FIRST_REFLECTION_INFER_VS_SUMMARY_CHAT_INSTRUCTION
            self_reflection_few_shot = FIRST_REFLECTION_INFER_VS_SUMMARY_FEW_SHOT
            system_content = self_reflection_chat_instruction
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
            response = model.generate_chat(messages=messages)
            print(f"RESPONSE!!!!!!: {response}")
            return response
        else:
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

            response = model.generate_chat(messages=messages)
            print(f"RESPONSE!!!!!!: {response}")
            return response
    else:
        # Non-chat model (completion-based)
        prompt = f"{self_reflection_completion_instruction}\n{func}\n\nExplanation:"
    return model.generate(prompt)


def generic_infer_specifications(
    problem_context: str,
    model: ModelBase,
    infer_specifications_chat_instruction: str,
    infer_specifications_few_shot: str,
) -> Union[str, List[str]]:
    if model.is_chat:
        system_content = f"{infer_specifications_chat_instruction}\n{infer_specifications_few_shot}"
        user_content = (
            f"[problem context]:\n{problem_context}\n\n[inferred specifications]:"
        )

        print_messages(system_content, user_content)

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]

        response = model.generate_chat(messages=messages)
        return response
    
def generic_generate_code_summary(
    cur_func_impl: str,
    model: ModelBase,
    code_summary_chat_instruction: str,
    code_summary_few_shot: str,
) -> Union[str, List[str]]:
    if model.is_chat:
        system_content = f"{code_summary_chat_instruction}\n{code_summary_few_shot}"
        user_content = (
            f"[problem context]:\n{cur_func_impl}\n\n[Code Summary]:"
        )

        print_messages(system_content, user_content)

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]

        response = model.generate_chat(messages=messages)
        return response
