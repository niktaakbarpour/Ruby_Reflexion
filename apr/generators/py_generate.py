from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection

from typing import Optional, List, Union
import ast
import re
# from .parse import parse_code_block, add_code_block
from .rb_parse import parse_code_block, add_code_block


PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Ruby programming language writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write ONLY your full implementation in Ruby (restate the function signature but DO NOT write example usage).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Ruby programming language writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Ruby programming language code block to write your response. For example:\n```ruby\nputs 'Hello world!'\n```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with Ruby programming language code, NOT ENGLISH and NOT PYTHON. You will be given a buggy code implementation and its docstring by the user. Write ONLY your full correct implementation in Ruby (DO NOT write example usage)."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only Ruby programming language code. You will be given a buggy code implementation and its docstring by the user. Write your full correct implementation in Ruby."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Ruby programming language assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation in Ruby."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Ruby programming language assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation in Ruby."
PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```ruby
def strlen(string)\n
    # Return length of given string\n
    string.chars.map(&:ord).sum\n
end
```

[unit test results from previous impl]:
Tests passed:
assert_equal 0, (->(string) { strlen(string) }).call("")

Tests failed:
assert_equal 3, (->(string) { strlen(string) }).call("abc")

[reflection on previous impl]:
I realized that the implementation of strlen was incorrect because it summed the ASCII values of the characters instead of simply returning the length of the string, which caused the test failures. My plan for improving the result is to modify the strlen function to return the length of the string using string.length, which will pass the test cases that expect the correct character count.

[improved impl]:
```ruby
def strlen(string)\n
    # Return length of given string\n
    string.length\n
end
```
END EXAMPLES

'''

PY_REFLEXION_FEW_SHOT = '''Example 1:
[previous impl]:
```ruby
def strlen(string)\n
    # Return length of given string\n
    string.chars.map(&:ord).sum\n
end
```

[unit test results from previous impl]:
Tests passed:
assert_equal 0, (->(string) { strlen(string) }).call("")

Tests failed:
assert_equal 3, (->(string) { strlen(string) }).call("abc")

[reflection on previous impl]:
I realized that the implementation of strlen was incorrect because it summed the ASCII values of the characters instead of simply returning the length of the string, which caused the test failures. My plan for improving the result is to modify the strlen function to return the length of the string using string.length, which will pass the test cases that expect the correct character count.

[improved impl]:
```ruby
def strlen(string)\n
    # Return length of given string\n
    return string.length\n
end
```
END EXAMPLES

'''
PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."
PY_SELF_REFLECTION_FEW_SHOT = """Example 1:
[function impl]:
```ruby
def palindrome?(string)\n
    return string.downcase == string.downcase.reverse\n
end
```
[unit test results from previous impl]:

Tests passed:
assert_equal true, (->(string) { palindrome?(string) }).call("madam")
assert_equal true, (->(string) { palindrome?(string) }).call("abcba")

Tests failed:
assert_equal false, (->(string) { palindrome?(string) }).call("hello") # output: true

[reflection on previous impl]: I realized that the implementation of palindrome? was incorrect because it was comparing the string directly to its reversed version without checking for case sensitivity or non-alphabetic characters, which caused the test failures. My plan for improving the result is to ensure that both the string and its reversed version are checked after removing non-alphabetic characters and converting to lowercase, which will pass the test cases that expect the correct palindrome check.

Example 2:
[function impl]:
```ruby
def longest_subarray_with_sum_limit(arr, limit)\n
    return arr.each_with_index.max_by { |_, idx| arr[0..idx].sum <= limit ? idx + 1 : 0 }&.last || 0\n
end

```
[unit test results from previous impl]:

Tests passed:
assert_equal true, (->(string) { palindrome?(string) }).call("madam")
assert_equal true, (->(string) { palindrome?(string) }).call("abcba")

Tests failed:
assert_equal false, (->(string) { palindrome?(string) }).call("hello") # output: true

[reflection on previous impl]: I realized that the implementation of longest_subarray_with_sum_limit was incorrect because I was using each_with_index.max_by to find the longest subarray, which did not properly calculate the maximum subarray length under the sum limit constraint. My plan for improving the result is to change the approach to use a sliding window technique, which will efficiently find the longest subarray with the sum constraint and handle the edge cases properly.
END OF EXAMPLES
"""

PY_TEST_GENERATION_FEW_SHOT = """Examples:
func signature:
def strlen(string)\n
    # Return length of given string
    # This function takes a string as input and returns the number of characters in the string.
unit tests:
[
    {
      "input": "abc\r\n",
      "output": [
        "3"
      ]
    },
    {
      "input": "hello world\r\n",
      "output": [
        "11"
      ]
    },
    {
      "input": "ruby\r\n",
      "output": [
        "4"
      ]
    },
    {
      "input": "12345\r\n",
      "output": [
        "5"
      ]
    },
]
"""

PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI Ruby programming language coding assistant that can writenew, unique, diverse, and intuitive Ruby test cases for codes given the docstring. In this step you should only generate sample input and output not function implemention and not test suite. Do not rewrite the tests that are already in the benchmark.

{PY_TEST_GENERATION_FEW_SHOT}"""

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI Ruby programming language coding assistant that can write new, unique, diverse, and intuitive Ruby test cases for codes given the docstring. In this step you should only generate sample input and output not function implemention and not test suite. Do not rewrite the tests that are already in the benchmark."""



class PyGenerator(Generator):
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "ruby"),
            self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT
        )

    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        return generic_generate_func_impl(
            func_sig=func_sig,
            model=model,
            strategy=strategy,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            reflexion_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
            reflexion_few_shot=PY_REFLEXION_FEW_SHOT_ADD,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "ruby"),
            add_code_block=lambda x: add_code_block(x, "ruby"),
        )

    def internal_tests(self, func_sig: str, model: ModelBase, max_num_tests: int = 5) -> List[str]:
        def parse_tests(tests: List[str]) -> List[str]:
            return [test.strip() for test in tests if "assert_equal" in test.strip()]
        """
        Generates tests for a function.
        """
        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=PY_TEST_GENERATION_FEW_SHOT,
            test_generation_chat_instruction=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            test_generation_completion_instruction=PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
            # parse_tests=parse_tests,
            # is_syntax_valid=py_is_syntax_valid,
            # is_syntax_valid=rb_is_syntax_valid,
        )

DUMMY_FUNC_SIG = "def dummy_func\n"  # Ruby style function declaration
DUMMY_FUNC_CALL = "dummy_func\nend"  # Added end keyword for Ruby function

# DUMMY_FUNC_SIG = "def func\nend"
# DUMMY_FUNC_CALL = "func"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)

# def py_fix_indentation(func_body: str) -> str:
#     """
#     Fix indentation for Ruby code. Ruby typically uses 2 spaces for indentation,
#     and does not rely on indentation for syntax like Ruby does.
#     """
#     func_body = fix_turbo_response(func_body)
    
#     def convert_indentation_to_ruby(code: str) -> str:
#         lines = code.splitlines()
#         result_lines = []
#         indent_level = 0
        
#         for line in lines:
#             stripped_line = line.lstrip()
            
#             # Decrease indent level for 'end', 'else', 'elsif' keywords
#             if stripped_line.startswith(('end', 'else', 'elsif')):
#                 indent_level = max(0, indent_level - 1)
            
#             # Add proper indentation
#             if stripped_line:  # Only indent non-empty lines
#                 result_lines.append('  ' * indent_level + stripped_line)
#             else:
#                 result_lines.append('')
            
#             # Increase indent level after certain keywords
#             if stripped_line.startswith(('def ', 'class ', 'module ', 'if ', 'unless ',
#                                        'case ', 'while ', 'until ', 'begin ', 'do ',
#                                        'else', 'elsif')) and not stripped_line.endswith('end'):
#                 indent_level += 1
            
#             # Handle one-line blocks with do
#             if 'do |' in stripped_line or 'do' == stripped_line.rstrip()[-2:]:
#                 indent_level += 1
            
#             # Decrease indent level for each 'end' keyword in the line
#             if stripped_line.count('end') > 0:
#                 # Only count 'end' keywords that are not part of other words
#                 ends = sum(1 for word in stripped_line.split() if word == 'end')
#                 indent_level = max(0, indent_level - ends)
        
#         return '\n'.join(result_lines)

#     # First, fix any markdown code block syntax
#     func_body = fix_markdown(func_body)
    
#     # Then apply Ruby-style indentation
#     properly_indented = convert_indentation_to_ruby(func_body)
    
#     # Verify the syntax is valid with our Ruby syntax checker
#     if rb_is_syntax_valid(f"{DUMMY_FUNC_SIG}{properly_indented}\n{DUMMY_FUNC_CALL}"):
#         return properly_indented
        
#     # If syntax check fails, return original with basic 2-space indentation
#     return func_body.strip()



# def py_fix_indentation(func_body: str) -> str:
#     func_body = fix_turbo_response(func_body)
#     """
#     3 cases:
#         1. good syntax
#         2. first line not good
#         3. entire body not good
#     """
#     def parse_indent_rec(f_body: str, cur_state: int) -> str:
#         f_body = fix_markdown(f_body)
#         if cur_state > 1:
#             return f_body
#         code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
#         try:
#             exec(code)
#             return f_body
#         except (IndentationError, SyntaxError):
#             p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
#             return parse_indent_rec(p_func(func_body), cur_state + 1)
#         except Exception:
#             return f_body
#     return parse_indent_rec(func_body, 0)

def rb_is_syntax_valid(code: str) -> bool:
    """
    A basic Ruby syntax checker implemented in pure Ruby.
    This checks for basic syntax rules like matching keywords and brackets.
    """
    code = code.strip()
    # Check for matching 'def' and 'end' keywords
    def_count = 0
    end_count = 0
    in_string = False
    string_char = None
    brackets = []  # Stack for (), [], {}

    lines = code.split('\n')
    for line in lines:
        line = line.strip()

        # Skip comments
        if line.startswith('#'):
            continue

        for i, char in enumerate(line):
            # Handle string literals
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue

            if in_string:
                continue

            # Check for 'def' and 'end' pairs
            if line[i:].startswith('def '):
                def_count += 1
            elif char == 'e' and line[i:].startswith('end') and (i+3 >= len(line) or not line[i+3].isalnum()):
                end_count += 1

            # Check brackets
            if char in '([{':
                brackets.append(char)
            elif char in ')]}':
                if not brackets:
                    return False
                last_open = brackets.pop()
                if (char == ')' and last_open != '(' or 
                    char == ']' and last_open != '[' or 
                    char == '}' and last_open != '{'):
                    return False

        # More flexible check for 'assert_equal'
        if line.startswith('assert_equal'):
            if '(' in line and ')' in line:
                # Rough check for valid method call with arguments
                inside_parens = line[line.index('(') + 1 : line.rindex(')')]
                if not inside_parens.strip():
                    return False  # Empty parentheses
            else:
                parts = line.split(None, 2)
                if len(parts) < 3:
                    return False  # Not enough parts in assert_equal

    # Final checks
    return (def_count == end_count and  # All 'def' have matching 'end'
            not brackets and             # All brackets are closed
            not in_string)               # No unclosed strings


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

