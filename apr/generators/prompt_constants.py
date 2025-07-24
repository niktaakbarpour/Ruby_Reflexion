PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Ruby programming language writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write ONLY your full implementation in Ruby (restate the function signature but DO NOT write example usage).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Ruby programming language writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Ruby programming language code block to write your response. For example:\n```ruby\nputs 'Hello world!'\n```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with Ruby programming language code, NOT ENGLISH and NOT PYTHON. You will be given a buggy code implementation and its docstring by the user. Write ONLY your full correct implementation in Ruby (DO NOT write example usage). In other words your task is automatic program repair."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only Ruby programming language code. You will be given a buggy code implementation and its docstring by the user. Write your full correct implementation in Ruby."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Ruby programming language assistant. You will be given your past function implementation, a series of unit tests, problem description, and a hint to change the implementation appropriately. Write your full implementation in Ruby based on the hint given to you (it tells you what was wrong in previous implementation), unit test results, and problem context."
RB_REFLEXION_CHAT_INSTRUCTION_SELF_OMIT = "You are an AI Ruby programming language assistant. You will be given problem contect, your past function implementation, and a series of unit tests. Write your full correct implementation based on problem requirements in Ruby."
RB_REFLEXION_CHAT_INSTRUCTION_TEST_OMIT = "You are an AI Ruby programming language assistant. You will be given problem docstring, your past function implementation and a hint to change the implementation appropriately. Write your full implementation in Ruby."
PY_FIRST_REFLEXION_CHAT_INSTRUCTION = "You are an AI Ruby programming language assistant. You will be given problem docstring, incorrect user function implementation, and a hint to change the implementation appropriately. Write your full implementation in Ruby."
RB_FIRST_REFLEXION_CHAT_INSTRUCTION_FIRST_OMIT = "You are an AI Ruby programming language assistant. You will be given problem context and incorrect user function implementation. Write your full correct implementation based on problem requirements in Ruby."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Ruby programming language assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation in Ruby."
RB_SCOT_CHAT_INSTRUCTION = '''
You are an expert Ruby programming assistant.

You will be given:
- Your past function implementation,
- A series of unit tests,
- Its docstring describing the intended behavior,
- And a hint to guide you toward the correct solution.

Your task:
- First, carefully plan the correct solution.
    - Write a rough structured plan using programming structures: **sequence**, **branch** (conditional logic), and **loop** (iteration).
    - Clearly specify the inputs, outputs, and main steps.
- Then, write the full corrected Ruby code according to your structured plan.

Always follow this two-step format:
1. Structured Plan
2. Corrected Ruby Code
'''

RB_FIRST_SCOT_CHAT_INSTRUCTION = '''
You are an expert Ruby programming assistant.

You will be given:
- An incorrect Ruby function implementation,
- Its docstring describing the intended behavior,
- And a hint to guide you toward the correct solution.

Your task:
- First, carefully plan the correct solution.
    - Write a rough structured plan using programming structures: **sequence**, **branch** (conditional logic), and **loop** (iteration).
    - Clearly specify the inputs, outputs, and main steps.
- Then, write the full corrected Ruby code according to your structured plan.

Always follow this two-step format:
1. Structured Plan
2. Corrected Ruby Code
'''

RB_REFLEXION_SCOT_FEW_SHOT_ADD = '''Example 1:
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

[Structured Chain-of-Thought]:
Input: Two integers, x and y.
Output: Four integers representing two triangle vertices (besides (0, 0)).

Step-by-step plan:
1. Read two integers x and y from user input.
2. Determine the quadrant based on the signs of x and y:
2.1. Sequence:
2.1.1. If x > 0 and y > 0:
2.1.1.1. Calculate L = x + y.
2.1.1.2. Output vertices: (0, L), (L, 0).
2.2. Branch:
2.2.1. If x > 0 and y < 0:
2.2.1.1. Calculate L = x - y. (since y is negative, subtracting makes L positive)
2.2.1.2. Output vertices: (0, -L), (L, 0) to ensure y-coordinate remains negative.
2.2.2. If x < 0 and y > 0:
2.2.2.1. Calculate L = y - x. (since x is negative, y-x is positive)
2.2.2.2. Output vertices: (-L, 0), (0, L).
2.2.3. If x < 0 and y < 0:
2.2.3.1. Calculate L = x + y. (both negative)
2.2.3.2. Output vertices: (L, 0), (0, L).
3. Loop: (No loop needed in this case as it is a single conditional dispatch.)

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

PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:

[problem context]:
Problem description: Vasily the bear has a favorite rectangle, it has one vertex at point (0, 0), and the opposite vertex at point (x, y). Of course, the sides of Vasya's favorite rectangle are parallel to the coordinate axes. Vasya also loves triangles, if the triangles have one vertex at point B = (0, 0). That's why today he asks you to find two points A = (x1, y1) and C = (x2, y2), such that the following conditions hold:  the coordinates of points: x1, x2, y1, y2 are integers. Besides, the following inequation holds: x1 &lt; x2;  the triangle formed by point A, B and C is rectangular and isosceles ( is right);  all points of the favorite rectangle are located inside or on the border of triangle ABC;  the area of triangle ABC is as small as possible. Help the bear, find the required points. It is not so hard to proof that these points are unique.
Input format: The first line contains two integers x, y ( - 109 ≤ x, y ≤ 109, x ≠ 0, y ≠ 0).
Output format: Print in the single line four integers x1, y1, x2, y2 — the coordinates of the required points.
A pre-run execution outcome of buggy source code: WRONG_ANSWER (The code compiles and runs but does not produce the correct output.)

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

RB_REFLEXION_FEW_SHOT_ADD_SELF_OMIT = '''Example 1:
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

RB_REFLEXION_FEW_SHOT_ADD_TEST_OMIT = '''Example 1:

[problem context]:
Problem description: Vasily the bear has a favorite rectangle, it has one vertex at point (0, 0), and the opposite vertex at point (x, y). Of course, the sides of Vasya's favorite rectangle are parallel to the coordinate axes. Vasya also loves triangles, if the triangles have one vertex at point B = (0, 0). That's why today he asks you to find two points A = (x1, y1) and C = (x2, y2), such that the following conditions hold:  the coordinates of points: x1, x2, y1, y2 are integers. Besides, the following inequation holds: x1 &lt; x2;  the triangle formed by point A, B and C is rectangular and isosceles ( is right);  all points of the favorite rectangle are located inside or on the border of triangle ABC;  the area of triangle ABC is as small as possible. Help the bear, find the required points. It is not so hard to proof that these points are unique.
Input format: The first line contains two integers x, y ( - 109 ≤ x, y ≤ 109, x ≠ 0, y ≠ 0).
Output format: Print in the single line four integers x1, y1, x2, y2 — the coordinates of the required points.
A pre-run execution outcome of buggy source code: WRONG_ANSWER (The code compiles and runs but does not produce the correct output.)

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

[reflection on previous impl]: This implementation is incorrect because in the case where x is positive and y is negative, the implementation calculates the coordinate using x - y, which produces a positive value. However, the expected output requires the y-coordinate of one triangle vertex to be negative so that the triangle properly encloses the rectangle. In short, the sign for the y-coordinate is handled incorrectly in this quadrant.

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


[Structured Chain-of-Thought]:
Input: Array of numbers
Output: Printed formatted string showing total sum and all numbers.

Step-by-step plan:
1. Initialize total ← 0 and formatted_values ← empty list.
2. For each element val in values:
2.1. Add val to total.
2.2. Append val (as string) to formatted_values list.
3. After the loop, join formatted_values with commas.
4. Print "Total: <total>, Values: <formatted_values>".

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


[Structured Chain-of-Thought]:
Input: User's password attempt (string)
Output: Printed authentication message.

Step-by-step plan:
1. Compare attempt with password.
2. If they are equal:
2.1. Print "Access granted".
3. Else:
3.1 Print "Access denied".

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


[Structured Chain-of-Thought]:
Input: 2D array (matrix)
Output: Printed sum of diagonal elements.

Step-by-step plan:
1. Initialize sum ← 0, i ← 0, j ← 0.
2. While i < number of rows and j < number of columns:
2.1. Add matrix[i][j] to sum.
2.2. Increment both i and j to move along the diagonal.
3. After loop ends, print sum.

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

RB_FIRST_REFLEXION_FEW_SHOT_ADD_FIRST_OMIT = '''Example 1:
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

FIRST_REFLECTION_INFER_VS_SUMMARY_CHAT_INSTRUCTION = """You are a helpful Ruby programming assistant. You will be given a code summary and an inferred specification of a programming problem. Your task is to analyze both and explain, in concise natural language, the differences between the code summary and the inferred specification.
Describe how the user should adjust their code so the code summary aligns with the inferred specification, but do not provide or suggest any corrected code.
Your response should be 2–3 sentences, focusing only on what is wrong. No code will be provided, only natural language descriptions.
"""

PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Ruby programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
RB_SELF_REFLECTION_CHAT_INSTRUCTION_TEST_OMIT = "You are a Ruby programming assistant. You will be given a problem context and a function implementation. Your goal is to write a few sentences to explain why your implementation is wrong based on program requirements. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
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

RB_SELF_REFLECTION_FEW_SHOT_TEST_OMIT = """Example 1:
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

[reflection on previous impl]: This is not correct because in the case where x is positive and y is negative, the implementation calculates the coordinate using x - y, which produces a positive value. However, the expected output requires the y-coordinate of one triangle vertex to be negative so that the triangle properly encloses the rectangle. In short, the sign for the y-coordinate is handled incorrectly in this quadrant.

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

FIRST_REFLECTION_INFER_VS_SUMMARY_FEW_SHOT = '''Examples:
[Code Summary]:
Function Intent:
    This code reads two integers x and y from input and prints four numbers forming
    the coordinates of a rectangle or line segment, based on the signs of x and y.
Input Assumptions:
    - The input is a single line containing two integers separated by a space.
Expected Behavior:
    - If both x and y are positive, it prints "0 (x+y) (x+y) 0".
    - If x > 0 and y < 0, it prints "0 (x-y) (x-y) 0".
    - If y > 0 (and x ≤ 0), it prints "-(y-x) 0 0 (y-x)".
    - Otherwise, it prints "(x+y) 0 0 (x+y)".
Edge Cases:
    - When x = 0 or y = 0, it falls into the third or fourth condition depending
      on the sign, which may result in zero or negative values in the output.

[Inferred Specification]:
Function Intent:
    The function is intended to determine the coordinates of two points A = (x1, y1)
    and C = (x2, y2) that define a right isosceles triangle ABC (with B = (0, 0)) 
    such that the rectangle with vertices (0, 0) and (x, y) lies entirely inside or 
    on the boundary of this triangle. The triangle should have the smallest possible area.
Input Assumptions:
    - The input consists of two integers x and y, representing the coordinates of 
      the opposite vertex of the rectangle (0, 0) and (x, y).
    - x and y are non-zero integers, and their range is -10^9 ≤ x, y ≤ 10^9.
Expected Behavior:
    - The function should output four integers x1, y1, x2, y2, which are the 
      coordinates of points A and C.
    - The output should ensure:
        1. x1 < x2.
        2. The triangle formed by A, B, and C is right-angled and isosceles.
        3. The rectangle lies entirely within or on the boundary of triangle ABC.
        4. The area of triangle ABC is minimized.
Edge Cases:
    - Large positive or negative values of x and y, ensuring that the solution 
      correctly accounts for all quadrants.
    - When x and y have different signs (rectangle lies across quadrants).
    - Minimal cases where |x| = 1 or |y| = 1, requiring careful selection of A and C.

[self-reflection]:
The code summary describes logic that prints coordinates based solely on the signs 
and sums of x and y, but it does not address the problem of finding points A and C 
that form a right isosceles triangle containing the rectangle. The inferred specification 
requires geometric reasoning to ensure the triangle is both right-angled and minimal 
in area while containing the rectangle, which the current code does not implement. 
The user should adjust the logic to calculate x1, y1, x2, and y2 based on the rectangle’s 
dimensions and orientation rather than relying on simple arithmetic with x and y.
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

RB_TEST_GENERATION_EDGE_FEW_SHOT = """Examples:
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
        "output": ["0 15 15 0"]
    },
    {
        "input": "-10 5\r\n",
        "output": ["-15 0 0 15"]
    },
    {
        "input": "1 -1\r\n",
        "output": ["0 -2 2 0"]
    },
    {
        "input": "-1 -1\r\n",
        "output": ["-2 0 0 -2"]
    },
    {
        "input": "-10 -1000000000\r\n",
        "output": ["-1000000010 0 0 -1000000010"]
    },
    {
        "input": "1000000000 1000000000\r\n",
        "output": ["0 2000000000 2000000000 0"]
    }
]
"""

RB_TEST_GENERATION_IO_COT_FEW_SHOT = """Examples:
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


[Chain-of-Thought Reasoning]:

**Step 1: Generate Input**
- I need to choose a valid input (x, y), where both x and y are non-zero integers.
- Let's pick x = 10 and y = 5.
- So the input is `"10 5\\r\\n"`.

**Step 2: Generate Expected Output**
- Since x > 0 and y > 0, the point lies in the first quadrant.
- To enclose the rectangle inside an isosceles triangle rooted at (0, 0), the triangle legs should extend to x + y on both axes.
- So, (x1, y1) = (0, x + y) = (0, 15), and (x2, y2) = (x + y, 0) = (15, 0).
- Therefore, the correct output is `["0 15 15 0"]`.


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

RB_TEST_VALIDATION_IO_COT_FEW_SHOT = """Examples:
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

[Test case to validate]:
{
  "input": "20 -10\\r\\n",
  "output": ["0 30 30 0"]
}

[Chain-of-Thought Validation]:

1. **Input Review**:
   - Input: x = 20, y = -10
   - x > 0 and y < 0 → the point lies in the fourth quadrant.

2. **Expected Geometry**:
   - We need a triangle that fully encloses the rectangle with corners (0,0) to (20, -10).
   - To ensure coverage and symmetry in an isosceles right triangle:
     - Let L = x - y = 20 - (-10) = 30
     - A = (0, -30), C = (30, 0)

3. **Expected Output**:
   - Triangle vertices: A = (0, -30), C = (30, 0)
   - Output: ["0 -30 30 0"]

4. **Compare with Provided Output**:
   - Provided: ["0 30 30 0"] ← ⚠️ y-coordinate is incorrectly positive

❌ Incorrect output

✅ Corrected test case:
{
  "input": "20 -10\\r\\n",
  "output": ["0 -30 30 0"]
}

[unit tests]:
[
    {
      "input": "20 -10\\r\\n",
      "output": [
        "0 -30 30 0"
      ]
    },
]
"""

PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI Ruby programming language coding assistant that can write new, unique, diverse, and intuitive Ruby test cases for codes given the docstring. In this step you should only generate sample input and output not function implemention and not test suite.

{PY_TEST_GENERATION_FEW_SHOT}"""

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI Ruby programming language coding assistant that can write new, unique, diverse, and intuitive Ruby test cases for codes given the problem context containing following information:
- The buggy source code provided.
- The problem description, which explains the intended behavior of the program.
- The input format, which describes the structure, range, and constraints of inputs.
- The expected output format, which specifies how the program's output should be structured.
- The pre-run execution outcome, which describes how the buggy code currently behaves.

In this step you should only generate sample input and output not function implemention and not test suite."""

RB_TEST_GENERATION_EDGE_CHAT_INSTRUCTION = """You are an AI Ruby programming language coding assistant tasked with generating high-quality test cases based on the provided problem context, which includes:
- The problem description, which explains the intended behavior of the program,
- The input format, which describes the structure, range, and constraints of inputs,
- The expected output format, which specifies how the program's output should be structured, and
- The pre-run execution outcome, which describes how the buggy code currently behaves.

**Role**: As a tester, your goal is to create comprehensive and diverse test cases that evaluate the correctness and robustness of the Ruby function under various scenarios.

Create two test cases for each of these three categories:

**1. Basic Test Cases**:
- **Objective**: Verify the function’s fundamental correctness under standard conditions.
- Include typical input values expected from everyday use.

**2. Edge Test Cases**:
- **Objective**: Challenge the function with extreme, unusual, or minimal input values.
- These should test boundary conditions and rare cases that could reveal hidden bugs.

**3. Large Scale Test Cases**:
- **Objective**: Evaluate the function’s behavior and performance under large or computationally intensive inputs.
- Ensure the function handles high-volume data without crashing or slowing down.

**Instructions**:
- Generate all test cases as a list of dictionaries, each containing an \"input\" and an \"output\".
- Do NOT include the function implementation or a full test suite. Only return test inputs and expected outputs.
- Return the test cases as valid JSON array.

Your response should be limited to a structured JSON list of test cases only, without additional explanation or prose."""

RB_TEST_GENERATION_IO_CHAT_INSTRUCTION = """You are an AI Ruby programming language coding assistant tasked with generating high-quality test cases based on the provided problem context, which includes:
- The buggy source code,
- The problem description, which explains the intended behavior of the program,
- The input format, which describes the structure, range, and constraints of inputs,
- The expected output format, which specifies how the program's output should be structured, and
- The pre-run execution outcome, which describes how the buggy code currently behaves.

**Role**: You are a test engineer. Your job is to create diverse and valid test cases to evaluate the function's correctness and robustness.

Each test case must be generated in two phases:
1. **Input Generation**: Based on the input format and problem requirements, generate a meaningful and valid input.
2. **Output Derivation**: Given the generated input and the intended behavior described in the problem statement, use chain of thoughts to infer the expected correct output. Do not replicate the buggy behavior — your output must reflect the correct solution.

**Instructions**:
- Return a list of dictionaries in JSON format. Each item must include:
  - `"input"`: a single string input (use newline characters if needed),
  - `"output"`: a list of output lines (as strings),

- Do **not** include function implementations or test harnesses.
- Ensure all inputs are valid and all outputs match the intended behavior of the problem.
- Do not output any extra explanation — only the test cases in structured format."""

RB_TEST_GENERATION_IO_COT_CHAT_INSTRUCTION = """You are an AI Ruby programming language coding assistant tasked with generating high-quality test cases based on the provided problem context, which includes:
- The buggy source code,
- The problem description, which explains the intended behavior of the program,
- The input format, which describes the structure, range, and constraints of inputs,
- The expected output format, which specifies how the program's output should be structured, and
- The pre-run execution outcome, which describes how the buggy code currently behaves.

**Role**: You are a test engineer. Your job is to create diverse and valid test cases to evaluate the function's correctness and robustness.

Each test case must be generated in two phases:
1. **Input Generation**: Based on the input format and problem requirements, generate a meaningful and valid input.
2. **Output Derivation**: Given the generated input and the intended behavior described in the problem statement, infer the expected correct output. Do not replicate the buggy behavior — your output must reflect the correct solution.

**Important**: Think step-by-step during both phases. First decide the input. Then explain how to derive the output. Then provide the final test case.

**Instructions**:
- Return a list of dictionaries in JSON format. Each item must include:
  - `"input"`: a single string input (use newline characters if needed),
  - `"output"`: a list of output lines (as strings),

- Do **not** include function implementations or test harnesses.
- Ensure all inputs are valid and all outputs match the intended behavior of the problem.
- Do not output any extra explanation — only the test cases in structured format."""

RB_TEST_VALIDATION_IO_CHAT_INSTRUCTION = """You are an AI Ruby programming language assistant tasked with validating the correctness of test cases based on the provided problem context, which includes:
- The buggy source code,
- The problem description, which explains the intended behavior of the program,
- The input format, which describes the structure, range, and constraints of inputs,
- The expected output format, which specifies how the program's output should be structured, and
- The pre-run execution outcome, which describes how the buggy code currently behaves.

**Role**: You are a validation agent. Your job is to check whether a given test case contains the correct **expected output** for a given **input**, based solely on the intended behavior described in the problem context.

**Instructions**:
- For each test case, verify that the input conforms to the input specification and that the expected output matches the correct output.
- Do not provide step-by-step reasoning or explain your thought process.
- Directly return your verdict.

**Your response must include**:
- A final decision: “✅ Correct output” or “❌ Incorrect output”.
- If incorrect, provide the corrected test case in the same JSON format (with both `"input"` and corrected `"output"`).

- Do not refer to or describe the buggy implementation.
- Only consider the intended correct behavior described in the problem.
"""

RB_TEST_VALIDATION_IO_COT_CHAT_INSTRUCTION = """You are an AI Ruby programming language assistant tasked with validating the correctness of test cases based on the provided problem context, which includes:
- The buggy source code,
- The problem description, which explains the intended behavior of the program,
- The input format, which describes the structure, range, and constraints of inputs,
- The expected output format, which specifies how the program's output should be structured, and
- The pre-run execution outcome, which describes how the buggy code currently behaves.

**Role**: You are a validation agent. Your job is to check whether a given test case contains the correct **expected output** for a given **input**, based on the intended behavior (not the buggy implementation).

Each test case will be validated in two steps:
1. **Input Review**: Read and understand the input string and verify that it is valid and meaningful according to the input specification.
2. **Output Reasoning**: Think step-by-step to derive the correct output based on the given input and the problem description. Then compare it to the provided output.

**Important**:
- Think out loud during your reasoning. Start by analyzing the input.
- Then explain how to derive the output.
- Finally, state whether the provided output is correct or incorrect, and why.

**Instructions**:
- Your response should include:
  - Step-by-step reasoning,
  - A final verdict: “✅ Correct output” or “❌ Incorrect output”,
  - If the output is incorrect, provide the corrected test case in the same JSON format (with both `"input"` and corrected `"output"`).

- Do not rerun or describe the buggy implementation.
- Base your validation purely on the expected behavior described in the problem context.
"""

RB_INFER_SPECIFICATIONS_CHAT_INSTRUCTION = """You are a software assistant helping to understand what a function is supposed to do.

[Task]
Given the problem description and other contents, infer and describe what the function is intended to do. Focus only on the expected intent and behavior based on the context.

[Instructions]
- Write a natural language specification that includes:
    - Function Intent: What the function is supposed to accomplish
    - Input Assumptions: What inputs are expected or assumed
    - Expected Behavior: What the function should return or perform
    - Edge Cases: Any special input conditions to consider
"""

RB_INFER_SPECIFICATIONS_FEW_SHOT = """[Example 1]
Problem description: Given an integer n, print the sum of the first n positive integers.

Input format: A single integer n (1 ≤ n ≤ 10^6)

Output format: A single integer representing the sum.

Time limit: 1 second

Memory limit: 256 MB

A pre-run execution outcome of buggy source code: WRONG_ANSWER (The code runs but produces incorrect output.)

Inferred Specification:
Function Intent: Compute the sum of all positive integers from 1 to n.
Input Assumptions: The input is a single integer n where 1 ≤ n ≤ 10^6.
Expected Behavior: Return the value of 1 + 2 + ... + n.
Edge Cases: n = 1 should return 1.

---

[Example 2]
Problem description: Given two integers a and b, compute their greatest common divisor (GCD).

Input format: Two integers a and b (1 ≤ a, b ≤ 10^9)

Output format: A single integer representing the GCD of a and b.

Time limit: 1 second

Memory limit: 256 MB

A pre-run execution outcome of buggy source code: TIME_LIMIT_EXCEEDED (The code takes too long to run.)

Inferred Specification:
Function Intent: Find the greatest common divisor of two integers.
Input Assumptions: Two integers a and b, both ≥ 1.
Expected Behavior: Return the largest integer that divides both a and b without remainder.
Edge Cases: a = b, a = 1 or b = 1, a much larger than b.
"""

RB_CODE_SUMMARY_CHAT_INSTRUCTION = """You are a software assistant helping to understand what a function is doing.
[Task]
Given the code, describe what the code is doing.

[Instructions]
- Write a natural language summary that includes:
    - Function Intent: What the function is supposed to accomplish
    - Input Assumptions: What inputs are expected
    - Expected Behavior: What the function should return or perform
    - Edge Cases: Any special input conditions to consider
"""

RB_CODE_SUMMARY_FEW_SHOT = """[Example 1]
Code:
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

Code Summary:
Function Intent: Determine the rectangle coordinates based on the quadrant of point (x, y).
Input Assumptions: Two integers x and y from standard input.
Expected Behavior: If (x, y) is in Q1: output 0 x+y x+y 0. If Q4, output 0 x-y x-y 0, If Q2, output -(y-x) 0 0 y-x. Else (Q3), output x+y 0 0 x+y.
Edge Cases: Handles x=0 or y=0 via the last two branches; sums/differences may be negative.

---

[Example 2]
Code:
```ruby
password = "secure123"
attempt = gets.chomp

if password == attempt
  puts "Access granted"
if password != attempt
  puts "Access denied"
end
```

Code Summary:
Check if the user’s input matches the password and print access status.
Input Assumptions: A fixed password string ("secure123"). A user input string (attempt).
Expected Behavior: If attempt matches password, prints "Access granted". If attempt does not match, prints "Access denied".
Edge Cases: The code has a bug: the second if should be elsif or a separate if with a closing end for the first condition, otherwise it may cause a syntax error.
"""