from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_first_reflection, generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection

from typing import Optional, List, Union
import ast
import re
# from .parse import parse_code_block, add_code_block
from .rb_parse import parse_code_block, add_code_block


PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Ruby programming language writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write ONLY your full implementation in Ruby (restate the function signature but DO NOT write example usage).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Ruby programming language writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Ruby programming language code block to write your response. For example:\n```ruby\nputs 'Hello world!'\n```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with Ruby programming language code, NOT ENGLISH and NOT PYTHON. You will be given a buggy code implementation and its docstring by the user. Write ONLY your full correct implementation in Ruby (DO NOT write example usage). In other words your task is automatic program repair."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only Ruby programming language code. You will be given a buggy code implementation and its docstring by the user. Write your full correct implementation in Ruby."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Ruby programming language assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation in Ruby."
RB_FIRST_SCOT_CHAT_INSTRUCTION = "You are an AI Ruby programming language assistant. You will be given incorrect user function implementation, its docstring, and a hint to change the implementation appropriately. Your task is to write the correct implementation in Ruby. You should first write a rough problem-solving process using three programming structures (i.e. sequential, branch, and loop structures) and then output the final code."
PY_FIRST_REFLEXION_CHAT_INSTRUCTION = "You are an AI Ruby programming language assistant. You will be given incorrect user function implementation and a hint to change the implementation appropriately. Write your full implementation in Ruby."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Ruby programming language assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation in Ruby."
PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```ruby
x, y = gets.split.map(&:to_i)

if x > 0 && y > 0
  # First quadrant: both coordinates positive.
  # Triangle vertices: (0, 0), (0, x+y), (x+y, 0)
  puts "0 #{x+y} #{x+y} 0"
elsif x > 0 && y < 0
  # Fourth quadrant: x positive, y negative.
  # Triangle vertices: (0, 0), (0, x-y), (x-y, 0)
  puts "0 #{x-y} #{x-y} 0"
elsif y > 0
  # Second quadrant: x negative, y positive.
  # Triangle vertices: (0, 0), (-(y-x), 0), (0, y-x)
  # (Note: here y-x is positive since x is negative.)
  puts "#{-(y-x)} 0 0 #{y-x}"
else
  # Third quadrant: both coordinates negative.
  # Triangle vertices: (0, 0), (x+y, 0), (0, x+y)
  puts "#{x+y} 0 0 #{x+y}"
end

```
[unit test results]:

Tests passed:
[
    {
      "input": "10 5\r\n",
      "output": [
        "0 15 15 0"
      ]
    },
    {
      "input": "-10 5\r\n",
      "output": [
        "-15 0 0 15"
      ]
    },
    {
      "input": "-10 -1000000000\r\n",
      "output": [
        "-1000000010 0 0 -1000000010"
      ]
    },
]

Tests failed:
[
    {
      "input": "20 -10\r\n",
      "output": [
        "0 30 30 0"
      ]
    },
]

[reflection on previous impl]: The error occurs because in the case where x is positive and y is negative, the implementation calculates the coordinate using x - y, which produces a positive value. However, the expected output requires the y-coordinate of one triangle vertex to be negative so that the triangle properly encloses the rectangle. In short, the sign for the y-coordinate is handled incorrectly in this quadrant.

[improved impl]:
```ruby
x, y = gets.split.map(&:to_i)

if x > 0 && y > 0
  # First quadrant: both positive.
  L = x + y
  puts "0 #{L} #{L} 0"
elsif x > 0 && y < 0
  # Fourth quadrant: x positive, y negative.
  L = x - y  # y is negative, so L is positive.
  puts "0 #{-L} #{L} 0"
elsif x < 0 && y > 0
  # Second quadrant: x negative, y positive.
  L = y - x  # x is negative, so L is positive.
  puts "#{-L} 0 0 #{L}"
else
  # Third quadrant: both negative.
  # Here, x+y is negative; it serves as the leg length (with proper sign).
  puts "#{x+y} 0 0 #{x+y}"
end
```
END EXAMPLES

'''

RB_FIRST_SCOT_FEW_SHOT = '''
Example 1:
[incorrect function impl]:
```ruby
values = [5, 10, 15]
total = 0
formatted_values = ""

values.each do |val|
  formatted_values = formatted_values + val.to_s + ", "
  total = val
end

puts "Total: #{total}, Values: #{formatted_values}"
```

[problem context]:
This code processes an array of numbers, attempting to compute their sum and create a formatted string representation (e.g., "Total: 30, Values: 5, 10, 15").

[self-reflection]:
total = val overwrites total in each iteration instead of accumulating.
formatted_values = formatted_values + val.to_s + ", " always appends a trailing comma, even at the last element.


[Structured Chain-of-Thought (Sequence Structure)]:
Let's think step by step:
Input: Array of numbers.
Output: Printed formatted string.

1. Initialize total ← 0 and formatted_values ← "".
2. For each element val in values, do:
3. Compute total ← total + val.
4. Convert val to string and append to formatted_values.
5. After loop, remove trailing comma from formatted_values.
6. Print "Total: <total>, Values: <formatted_values>".

[Repaired Code]:
```ruby
values = [5, 10, 15]
total = 0
formatted_values = []

values.each do |val|
  total += val  # Fix: Accumulate instead of overwrite
  formatted_values << val.to_s  # Fix: Use an array for correct formatting
end

puts "Total: #{total}, Values: #{formatted_values.join(', ')}"  # Fix: Remove trailing comma
```

Example 2:
[incorrect function impl]:
```ruby
password = "secure123"
attempt = gets.chomp

if password == attempt
  puts "Access granted"
if password != attempt
  puts "Access denied"
end
```

[problem context]:
This snippet checks user authentication. The expected behavior is: If the user enters the correct password, print "Access granted". Otherwise, print "Access denied".

[self-reflection]:
The if password == attempt block is not properly closed.
The second condition (if password != attempt) runs independently, so it always executes, printing both messages at once.


[Structured Chain-of-Thought (Branch Structure)]:
Let's think step by step:
Input: String (password attempt).
Output: Printed authentication message.

1. If password == attempt then:
2. Print "Access granted"
3. Else:
4. Print "Access denied"

[Repaired Code]:
```ruby
password = "secure123"
attempt = gets.chomp

if password == attempt
  puts "Access granted"
else  # Fix: Ensure mutually exclusive branches
  puts "Access denied"
end
```

Example 3:
[incorrect function impl]:
```ruby
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

sum = 0
i = 0
j = 0

while i < matrix.length && j < matrix[0].length
  sum += matrix[i][j]
  i += 1
end

puts sum
```

[problem context]:
This code aims to compute the sum of the diagonal elements of a matrix.

[self-reflection]:
while i < matrix.length && j < matrix[0].length ensures the loop exits before iterating all diagonals.
j is never incremented, meaning it always reads matrix[i][0] instead of following the diagonal.


[Structured Chain-of-Thought (Loop Structure)]:
Let's think step by step:
Input: 2D array.
Output: Printed diagonal sum.

1. Initialize sum ← 0, i ← 0, j ← 0.
2. While i < matrix.length && j < matrix[0].length do:
3. Compute sum ← sum + matrix[i][j].
4. Update i ← i + 1, j ← j + 1 to follow diagonal.
5. Print sum.

[Repaired Code]:
```ruby
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

sum = 0
i = 0
j = 0

while i < matrix.length && j < matrix[0].length
  sum += matrix[i][j]
  i += 1  # Fix: Move both indices to traverse the diagonal
  j += 1
end

puts sum  # Output: 15
```
END EXAMPLES
'''

PY_FIRST_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[incorrect function impl]:
```ruby
x,y=gets.split.map(&:to_i)
if x > 0 and y > 0
    puts "0 #{x+y} #{x+y} 0"
elsif x > 0 and y < 0
    puts "0 #{x-y} #{x-y} 0"
elsif y > 0
    puts "#{-(y-x)} 0 0 #{y-x}"
else
    puts "#{x+y} 0 0 #{x+y}"
end

```

[problem context]:
Problem description: Vasily the bear has a favorite rectangle, it has one vertex at point (0, 0), and the opposite vertex at point (x, y). Of course, the sides of Vasya's favorite rectangle are parallel to the coordinate axes. Vasya also loves triangles, if the triangles have one vertex at point B = (0, 0). That's why today he asks you to find two points A = (x1, y1) and C = (x2, y2), such that the following conditions hold:  the coordinates of points: x1, x2, y1, y2 are integers. Besides, the following inequation holds: x1 &lt; x2;  the triangle formed by point A, B and C is rectangular and isosceles ( is right);  all points of the favorite rectangle are located inside or on the border of triangle ABC;  the area of triangle ABC is as small as possible. Help the bear, find the required points. It is not so hard to proof that these points are unique.
Input format: The first line contains two integers x, y ( - 109 ≤ x, y ≤ 109, x ≠ 0, y ≠ 0).
Output format: Print in the single line four integers x1, y1, x2, y2 — the coordinates of the required points.
A pre-run execution outcome of buggy source code: WRONG_ANSWER (The code compiles and runs but does not produce the correct output.)

[self-reflection]:
The provided code does not work correctly because it incorrectly calculates the coordinates for points A, B, and C based on the input values of x and y. The code attempts to determine the coordinates of points A and C based on the conditions for a right-isosceles triangle with one vertex at the origin (0,0), but it fails to correctly compute the coordinates for A and C. Additionally, the conditions for the area of the triangle and the placement of the rectangle inside the triangle are not properly handled in the code.

[improved impl]:
```ruby
x, y = gets.split.map(&:to_i)

if x > 0 && y > 0
  # First quadrant: both positive.
  L = x + y
  puts "0 #{L} #{L} 0"
elsif x > 0 && y < 0
  # Fourth quadrant: x positive, y negative.
  L = x - y  # y is negative, so L is positive.
  puts "0 #{-L} #{L} 0"
elsif x < 0 && y > 0
  # Second quadrant: x negative, y positive.
  L = y - x  # x is negative, so L is positive.
  puts "#{-L} 0 0 #{L}"
else
  # Third quadrant: both negative.
  # Here, x+y is negative; it serves as the leg length (with proper sign).
  puts "#{x+y} 0 0 #{x+y}"
end
```
END EXAMPLES

'''

PY_REFLEXION_FEW_SHOT = '''Example 1:
[previous impl]:
```ruby
x, y = gets.split.map(&:to_i)

if x > 0 && y > 0
  # First quadrant: both coordinates positive.
  # Triangle vertices: (0, 0), (0, x+y), (x+y, 0)
  puts "0 #{x+y} #{x+y} 0"
elsif x > 0 && y < 0
  # Fourth quadrant: x positive, y negative.
  # Triangle vertices: (0, 0), (0, x-y), (x-y, 0)
  puts "0 #{x-y} #{x-y} 0"
elsif y > 0
  # Second quadrant: x negative, y positive.
  # Triangle vertices: (0, 0), (-(y-x), 0), (0, y-x)
  # (Note: here y-x is positive since x is negative.)
  puts "#{-(y-x)} 0 0 #{y-x}"
else
  # Third quadrant: both coordinates negative.
  # Triangle vertices: (0, 0), (x+y, 0), (0, x+y)
  puts "#{x+y} 0 0 #{x+y}"
end

```
[unit test results]:

Tests passed:
[
    {
      "input": "10 5\r\n",
      "output": [
        "0 15 15 0"
      ]
    },
    {
      "input": "-10 5\r\n",
      "output": [
        "-15 0 0 15"
      ]
    },
    {
      "input": "-10 -1000000000\r\n",
      "output": [
        "-1000000010 0 0 -1000000010"
      ]
    },
]

Tests failed:
[
    {
      "input": "20 -10\r\n",
      "output": [
        "0 30 30 0"
      ]
    },
]

[reflection on previous impl]: The error occurs because in the case where x is positive and y is negative, the implementation calculates the coordinate using x - y, which produces a positive value. However, the expected output requires the y-coordinate of one triangle vertex to be negative so that the triangle properly encloses the rectangle. In short, the sign for the y-coordinate is handled incorrectly in this quadrant.

[improved impl]:
```ruby
x, y = gets.split.map(&:to_i)

if x > 0 && y > 0
  # First quadrant: both positive.
  L = x + y
  puts "0 #{L} #{L} 0"
elsif x > 0 && y < 0
  # Fourth quadrant: x positive, y negative.
  L = x - y  # y is negative, so L is positive.
  puts "0 #{-L} #{L} 0"
elsif x < 0 && y > 0
  # Second quadrant: x negative, y positive.
  L = y - x  # x is negative, so L is positive.
  puts "#{-L} 0 0 #{L}"
else
  # Third quadrant: both negative.
  # Here, x+y is negative; it serves as the leg length (with proper sign).
  puts "#{x+y} 0 0 #{x+y}"
end

```
END EXAMPLES

'''

FIRST_REFLECTION_CHAT_INSTRUCTION = """You are a helpful Ruby programming assistant. You are helping a user debug a Ruby program. The user has written some code, but it has errors. You will help the user by analyzing problem context which contains the following information:
The buggy source code provided.
The problem description, which explains the intended behavior of the program.
The input format, which describes the structure, range, and constraints of inputs.
The expected output format, which specifies how the program's output should be structured.
The pre-run execution outcome, which describes how the buggy code currently behaves.
Based on this information, explain why the provided code does not work correctly in a concise natural language response. Focus only on describing the issue and do not suggest or generate a corrected implementation.
Your response should be a short explanation (2-3 sentences) of what is wrong. Do not include code in your response."""

PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."
PY_SELF_REFLECTION_FEW_SHOT = """Example 1:
[function impl]:
```ruby
x, y = gets.split.map(&:to_i)

if x > 0 && y > 0
  # First quadrant: both coordinates positive.
  # Triangle vertices: (0, 0), (0, x+y), (x+y, 0)
  puts "0 #{x+y} #{x+y} 0"
elsif x > 0 && y < 0
  # Fourth quadrant: x positive, y negative.
  # Triangle vertices: (0, 0), (0, x-y), (x-y, 0)
  puts "0 #{x-y} #{x-y} 0"
elsif y > 0
  # Second quadrant: x negative, y positive.
  # Triangle vertices: (0, 0), (-(y-x), 0), (0, y-x)
  # (Note: here y-x is positive since x is negative.)
  puts "#{-(y-x)} 0 0 #{y-x}"
else
  # Third quadrant: both coordinates negative.
  # Triangle vertices: (0, 0), (x+y, 0), (0, x+y)
  puts "#{x+y} 0 0 #{x+y}"
end

```
[unit test results]:

Tests passed:
[
    {
      "input": "10 5\r\n",
      "output": [
        "0 15 15 0"
      ]
    },
    {
      "input": "-10 5\r\n",
      "output": [
        "-15 0 0 15"
      ]
    },
    {
      "input": "-10 -1000000000\r\n",
      "output": [
        "-1000000010 0 0 -1000000010"
      ]
    },
]

Tests failed:
[
    {
      "input": "20 -10\r\n",
      "output": [
        "0 30 30 0"
      ]
    },
]

[reflection on previous impl]: The error occurs because in the case where x is positive and y is negative, the implementation calculates the coordinate using x - y, which produces a positive value. However, the expected output requires the y-coordinate of one triangle vertex to be negative so that the triangle properly encloses the rectangle. In short, the sign for the y-coordinate is handled incorrectly in this quadrant.

END OF EXAMPLES
"""

PY_FIRST_SELF_REFLECTION_FEW_SHOT = '''Examples:
[incorrect function impl]:
```ruby
x,y=gets.split.map(&:to_i)
if x > 0 and y > 0
    puts "0 #{x+y} #{x+y} 0"
elsif x > 0 and y < 0
    puts "0 #{x-y} #{x-y} 0"
elsif y > 0
    puts "#{-(y-x)} 0 0 #{y-x}"
else
    puts "#{x+y} 0 0 #{x+y}"
end

```

[problem context]:
Problem description: Vasily the bear has a favorite rectangle, it has one vertex at point (0, 0), and the opposite vertex at point (x, y). Of course, the sides of Vasya's favorite rectangle are parallel to the coordinate axes. Vasya also loves triangles, if the triangles have one vertex at point B = (0, 0). That's why today he asks you to find two points A = (x1, y1) and C = (x2, y2), such that the following conditions hold:  the coordinates of points: x1, x2, y1, y2 are integers. Besides, the following inequation holds: x1 &lt; x2;  the triangle formed by point A, B and C is rectangular and isosceles ( is right);  all points of the favorite rectangle are located inside or on the border of triangle ABC;  the area of triangle ABC is as small as possible. Help the bear, find the required points. It is not so hard to proof that these points are unique.
Input format: The first line contains two integers x, y ( - 109 ≤ x, y ≤ 109, x ≠ 0, y ≠ 0).
Output format: Print in the single line four integers x1, y1, x2, y2 — the coordinates of the required points.
A pre-run execution outcome of buggy source code: WRONG_ANSWER (The code compiles and runs but does not produce the correct output.)

[self-reflection]:
The provided code does not work correctly because it incorrectly calculates the coordinates for points A, B, and C based on the input values of x and y. The code attempts to determine the coordinates of points A and C based on the conditions for a right-isosceles triangle with one vertex at the origin (0,0), but it fails to correctly compute the coordinates for A and C. Additionally, the conditions for the area of the triangle and the placement of the rectangle inside the triangle are not properly handled in the code.

```
END EXAMPLES

'''

PY_TEST_GENERATION_FEW_SHOT = """Examples:
[buggy code]:
x,y=gets.split.map(&:to_i)
if x > 0 and y > 0
    puts "0 #{x+y} #{x+y} 0"
elsif x > 0 and y < 0
    puts "0 #{x-y} #{x-y} 0"
elsif y > 0
    puts "#{-(y-x)} 0 0 #{y-x}"
else
    puts "#{x+y} 0 0 #{x+y}"
end


[problem context]:
Problem description: Vasily the bear has a favorite rectangle, it has one vertex at point (0, 0), and the opposite vertex at point (x, y). Of course, the sides of Vasya's favorite rectangle are parallel to the coordinate axes. Vasya also loves triangles, if the triangles have one vertex at point B = (0, 0). That's why today he asks you to find two points A = (x1, y1) and C = (x2, y2), such that the following conditions hold:  the coordinates of points: x1, x2, y1, y2 are integers. Besides, the following inequation holds: x1 &lt; x2;  the triangle formed by point A, B and C is rectangular and isosceles ( is right);  all points of the favorite rectangle are located inside or on the border of triangle ABC;  the area of triangle ABC is as small as possible. Help the bear, find the required points. It is not so hard to proof that these points are unique.

Input format: The first line contains two integers x, y ( - 109 ≤ x, y ≤ 109, x ≠ 0, y ≠ 0).

Output format: Print in the single line four integers x1, y1, x2, y2 — the coordinates of the required points.

A pre-run execution outcome of buggy source code: WRONG_ANSWER (The code compiles and runs but does not produce the correct output.)


[unit tests]:
[
    {
      "input": "10 5\r\n",
      "output": [
        "0 15 15 0"
      ]
    },
    {
      "input": "-10 5\r\n",
      "output": [
        "-15 0 0 15"
      ]
    },
    {
      "input": "20 -10\r\n",
      "output": [
        "0 -30 30 0"
      ]
    },
    {
      "input": "-10 -1000000000\r\n",
      "output": [
        "-1000000010 0 0 -1000000010"
      ]
    },
]
"""

PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI Ruby programming language coding assistant that can write new, unique, diverse, and intuitive Ruby test cases for codes given the docstring. In this step you should only generate sample input and output not function implemention and not test suite.

{PY_TEST_GENERATION_FEW_SHOT}"""

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI Ruby programming language coding assistant that can write new, unique, diverse, and intuitive Ruby test cases for codes given the problem context containing following information:
The buggy source code provided.
The problem description, which explains the intended behavior of the program.
The input format, which describes the structure, range, and constraints of inputs.
The expected output format, which specifies how the program's output should be structured.
The pre-run execution outcome, which describes how the buggy code currently behaves.
In this step you should only generate sample input and output not function implemention and not test suite."""



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
    
    def first_reflection(self, problem_context: str, func: str, model: ModelBase) -> str:
        return generic_generate_first_reflection(
            problem_context=problem_context,
            func=func,
            model=model,
            self_reflection_chat_instruction=FIRST_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "ruby"),
            self_reflection_few_shot=PY_FIRST_SELF_REFLECTION_FEW_SHOT
        )

    def func_impl(
        self,
        problem_context: str,
        model: ModelBase,
        strategy: str,
        is_first_reflection: bool,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        return generic_generate_func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            reflexion_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
            first_reflexion_chat_instruction=PY_FIRST_REFLEXION_CHAT_INSTRUCTION,
            reflexion_few_shot=PY_REFLEXION_FEW_SHOT_ADD,
            first_reflexion_few_shot=PY_FIRST_REFLEXION_FEW_SHOT_ADD,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "ruby"),
            add_code_block=lambda x: add_code_block(x, "ruby"),
        )

    def internal_tests(self, problem_context: str, func: str, model: ModelBase, max_num_tests: int = 5) -> List[str]:
        def parse_tests(tests: List[str]) -> List[str]:
            return [test.strip() for test in tests if "assert_equal" in test.strip()]
        """
        Generates tests for a function.
        """
        return generic_generate_internal_tests(
            problem_context=problem_context,
            func=func,
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

