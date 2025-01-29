import re
from typing import Optional


def parse_code_block(string: str, lang: str) -> Optional[str]:
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    print(f"match: {match}")

    if match:
        print(f"match2: {match}")
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        print(f"match3: {match}")
        return match.group(1)

    return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "ruby", "Only ruby is supported for now"
    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    found_end = False
    
    # Find the first function definition
    for i, line in enumerate(code_lines):
        stripped_line = line.strip()
        if stripped_line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                # Found another function definition, break
                break
        elif stripped_line == "end" and def_i != -1:
            # Found the end of the current function
            last_i = i
            found_end = True
            break

    if def_i == -1 or not found_end:
        return None

    # Include both the def line and the end line
    return "\n".join(code_lines[def_i:last_i + 1])


def add_code_block(string: str, lang: str) -> str:
    return f"```{lang}\n{string}\n```"


if __name__ == "__main__":
    CODE = """
# Some random Ruby code
puts "hello"

def my_wonderful_func
  x = 1
  if x == 1
    return "one"
  else
    return "not one"
  end
end

def another_func
  puts "this should not be included"
end
"""
    print(parse_code_block(CODE, "ruby"))

    CODE = """def total_match(list1, list2)
  total_chars_1 = list1.join.length
  total_chars_2 = list2.join.length
  
  if total_chars_1 <= total_chars_2
    return list1
  else
    return list2
  end
end"""
    print(parse_code_block(CODE, "ruby"))