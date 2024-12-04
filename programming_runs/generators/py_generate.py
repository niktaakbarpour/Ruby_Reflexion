from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection

from typing import Optional, List, Union
import ast
import re
# from .parse import parse_code_block, add_code_block
from .rb_parse import parse_code_block, add_code_block


PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Ruby writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Ruby writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Ruby code block to write your response. For example:\n```Ruby\nprint('Hello world!\n')```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with Ruby code, NOT ENGLISH. You will be given a function implementation and one or more test cases by the user. Write the correct full implementation (restate the function signature)."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only Ruby code. You will be given a function implementation and one or more test cases by the user. Write the correct full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Ruby assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Ruby assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation (restate the function signature)."
PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```ruby
def matrix_multiply\n  n,m = gets.chomp.split(\" \").map { |e| e.to_i }\n\n  a = Array.new(n){Array.new(m,0)}\n  b = Array.new(m)\n  c = Array.new(n,0)\n\n  k = 0\n  while k <= n-1\n    a[k] = gets.chomp.split.map { |e| e.to_i }\n    k += 1\n  end\n\n  t = 0\n  while t <= m-1\n    b[t] = gets.to_i\n    t += 1\n  end\n\n  for i in 0..(n-1)\n    for e in 0..(m-1)\n      c[i] += a[i][e] * b[e]\n    end\n  end\n\n  result = \"\"\n  for i in 0..(n-1)\n    result += c[i].to_s + \"\\n\"\n  end\n  result\nend\n
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
matrix_multiply.call.must_equal("15\n7\n7\n") # output: "13\n6\n14\n"

[reflection on previous impl]:
The implementation failed to produce the expected output "15\n7\n7\n", instead outputting "13\n6\n14\n". Since there's no clear operator misuse or logic error visible in the code, we should add debug prints to verify that arrays A and B are being populated correctly from the input before multiplication occurs.

[improved impl]:
```ruby
def matrix_multiply\n n,m = gets.chomp.split(\" \").map { |e| e.to_i }\n\na = Array.new(n){Array.new(m,0)}\nb = Array.new(m)\nc = Array.new(n,0)\n\nk = 0\nwhile k <= n-1\na[k] = gets.chomp.split.map { |e| e.to_i }\nk += 1\nend\nt = 0\nwhile t <= m-1\nb[t] = gets.to_i\nt += 1\nend\n\nfor i in 0..(n-1)\n  for e in 0..(m-1)\n    c[i] += a[i][e] * b[e]\n  end\nend\nfor i in 0..(n-1)\n  puts c[i]\nend
```
'''

PY_REFLEXION_FEW_SHOT = '''Example 1:
[previous impl]:
```ruby
def apartment_building
 xs = Array.new(4) {
   Array.new(3) {
     Array.new(10) { 0 }
   }
 }

 gets.to_i.times do 
   b,f,r,v = gets.split(" ").map(&:to_i)
   xs[b - 1][f - 1][r - 1] += v
 end

 puts xs.map {|building| 
   building.map {|floor| floor.join(" ") }.join("\n")
 }.join("\n" + "#" * 19 + "\n")
end
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
apartment_building.call("3\n1 1 3 1\n3 2 4 7\n4 3 8 1").must_equal "0 0 1 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n###################\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n###################\n0 0 0 0 0 0 0 0 0 0\n0 0 0 7 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n###################\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 1 0 0\n"

[reflection on previous impl]:
The implementation failed to match the expected output format. The core logic for building the 3D apartment array and updating values works correctly, but there are two formatting issues: 1.Missing leading spaces before each line of numbers in the output 2.Using 19 '#' characters for the separator lines instead of the required 20

[improved impl]:
```ruby
def apartment_building
 xs = Array.new(4) {
   Array.new(3) {
     Array.new(10) { 0 }
   }
 }
 n = gets.to_i
 n.times do 
   b,f,r,v = gets.split(" ").map(&:to_i)
   xs[b - 1][f - 1][r - 1] += v
 end
 
 result = xs.map {|building| 
   building.map {|floor| " " + floor.join(" ") }.join("\n")
 }.join("\n" + "#" * 20 + "\n")
 
 result + "\n"
end
```
END EXAMPLES

'''
PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."
PY_SELF_REFLECTION_FEW_SHOT = """Example 1:
[function impl]:
```ruby
def three_d_array
 xs = Array.new(4) {
   Array.new(3) {
     Array.new(10) { 0 }
   }
 }

 gets.to_i.times do 
   b,f,r,v = gets.split(" ").map(&:to_i)
   xs[b - 1][f - 1][r - 1] += v
 end

 result = xs.map {|building| 
   building.map {|floor| floor.join(" ") }.join("\n")
 }.join("\n" + "#" * 19 + "\n")
 
 result + "\n"
end
```

[unit test results]:
Tests passing:

Tests failing:
apartment_building.call("3\n1 1 3 1\n3 2 4 7\n4 3 8 1").must_equal "0 0 1 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n###################\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n###################\n0 0 0 0 0 0 0 0 0 0\n0 0 0 7 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n###################\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 1 0 0\n"

[self-reflection]:
The implementation failed to match the expected output format. The core logic for building the 3D apartment array and updating values works correctly, but there are two formatting issues: 1.Missing leading spaces before each line of numbers in the output 2.Using 19 '#' characters for the separator lines instead of the required 20

END OF EXAMPLES
"""

PY_TEST_GENERATION_FEW_SHOT = """Examples:
func signature:
def find_divsby3_and_has3
  n = gets.to_i
  for i in 1..n
    x = i
    if x % 3 == 0
      print " #{i}"
    else
      while x > 0
        if x % 10 == 3
          print " #{i}"
          break  
        end
        x /= 10
      end
    end
  end
  puts
end

unit tests:
find_divsby3_and_has3.call("94").must_equal " 3 6 9 12 13 15 18 21 23 24 27 30 31 32 33 34 35 36 37 38 39 42 43 45 48 51 53 54 57 60 63 66 69 72 73 75 78 81 83 84 87 90 93\n"
find_divsby3_and_has3.call("90").must_equal " 3 6 9 12 13 15 18 21 23 24 27 30 31 32 33 34 35 36 37 38 39 42 43 45 48 51 53 54 57 60 63 66 69 72 73 75 78 81 83 84 87 90\n"
find_divsby3_and_has3.call("206").must_equal " 3 6 9 12 13 15 18 21 23 24 27 30 31 32 33 34 35 36 37 38 39 42 43 45 48 51 53 54 57 60 63 66 69 72 73 75 78 81 83 84 87 90 93 96 99 102 103 105 108 111 113 114 117 120 123 126 129 130 131 132 133 134 135 136 137 138 139 141 143 144 147 150 153 156 159 162 163 165 168 171 173 174 177 180 183 186 189 192 193 195 198 201 203 204\n"
find_divsby3_and_has3.call("128").must_equal " 3 6 9 12 13 15 18 21 23 24 27 30 31 32 33 34 35 36 37 38 39 42 43 45 48 51 53 54 57 60 63 66 69 72 73 75 78 81 83 84 87 90 93 96 99 102 103 105 108 111 113 114 117 120 123 126\n"
find_divsby3_and_has3.call("166").must_equal " 3 6 9 12 13 15 18 21 23 24 27 30 31 32 33 34 35 36 37 38 39 42 43 45 48 51 53 54 57 60 63 66 69 72 73 75 78 81 83 84 87 90 93 96 99 102 103 105 108 111 113 114 117 120 123 126 129 130 131 132 133 134 135 136 137 138 139 141 143 144 147 150 153 156 159 162 163 165\n"
find_divsby3_and_has3.call("154").must_equal " 3 6 9 12 13 15 18 21 23 24 27 30 31 32 33 34 35 36 37 38 39 42 43 45 48 51 53 54 57 60 63 66 69 72 73 75 78 81 83 84 87 90 93 96 99 102 103 105 108 111 113 114 117 120 123 126 129 130 131 132 133 134 135 136 137 138 139 141 143 144 147 150 153\n"
"""

PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for the given functions.

{PY_TEST_GENERATION_FEW_SHOT}"""

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for the given functions."""


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
        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if ".must_equal" in test]
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
            parse_tests=parse_tests,
            # is_syntax_valid=py_is_syntax_valid,
            is_syntax_valid=rb_is_syntax_valid,
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

def py_fix_indentation(func_body: str) -> str:
    """
    Fix indentation for Ruby code. Ruby typically uses 2 spaces for indentation,
    and doesn't rely on indentation for syntax like Python does.
    """
    func_body = fix_turbo_response(func_body)
    
    def convert_indentation_to_ruby(code: str) -> str:
        lines = code.splitlines()
        result_lines = []
        indent_level = 0
        
        for line in lines:
            stripped_line = line.lstrip()
            
            # Decrease indent level for 'end', 'else', 'elsif' keywords
            if stripped_line.startswith(('end', 'else', 'elsif')):
                indent_level = max(0, indent_level - 1)
            
            # Add proper indentation
            if stripped_line:  # Only indent non-empty lines
                result_lines.append('  ' * indent_level + stripped_line)
            else:
                result_lines.append('')
            
            # Increase indent level after certain keywords
            if stripped_line.startswith(('def ', 'class ', 'module ', 'if ', 'unless ',
                                       'case ', 'while ', 'until ', 'begin ', 'do ',
                                       'else', 'elsif')) and not stripped_line.endswith('end'):
                indent_level += 1
            
            # Handle one-line blocks with do
            if 'do |' in stripped_line or 'do' == stripped_line.rstrip()[-2:]:
                indent_level += 1
            
            # Decrease indent level for each 'end' keyword in the line
            if stripped_line.count('end') > 0:
                # Only count 'end' keywords that are not part of other words
                ends = sum(1 for word in stripped_line.split() if word == 'end')
                indent_level = max(0, indent_level - ends)
        
        return '\n'.join(result_lines)

    # First, fix any markdown code block syntax
    func_body = fix_markdown(func_body)
    
    # Then apply Ruby-style indentation
    properly_indented = convert_indentation_to_ruby(func_body)
    
    # Verify the syntax is valid with our Ruby syntax checker
    if rb_is_syntax_valid(f"{DUMMY_FUNC_SIG}{properly_indented}\n{DUMMY_FUNC_CALL}"):
        return properly_indented
        
    # If syntax check fails, return original with basic 2-space indentation
    return func_body.strip()

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
    A basic Ruby syntax checker implemented in pure Python.
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
                
            # Count keywords outside of strings
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
