import re
from typing import Optional

def parse_code_block(string: str, lang: str) -> Optional[str]:
    import re

    if isinstance(string, list):
        string = string[0] if string else ""
    text = str(string)

    lines = text.splitlines()
    blocks = []

    open_re = re.compile(r'^[ \t]*(```|~~~)[ \t]*([^\n`]*)[ \t]*$')

    i = 0
    while i < len(lines):
        m = open_re.match(lines[i])
        if not m:
            # DEBUG: show any line that *looks* like a fence but didn’t match
            if '```' in lines[i] or '~~~' in lines[i]:
                print("OPEN didn’t match:", repr(lines[i]))
            i += 1
            continue

        fence, info = m.group(1), (m.group(2) or "")
        fence_char = fence[0]
        close_re = re.compile(rf'^[ \t]*{re.escape(fence)}[ \t]*$')

        j = i + 1
        while j < len(lines) and not close_re.match(lines[j]):
            # DEBUG for closing failures
            if (('```' in lines[j] or '~~~' in lines[j]) and
                not close_re.match(lines[j])):
                print("CLOSE didn’t match:", repr(lines[j]), "expected:", fence)
            j += 1

        if j >= len(lines):
            block = "\n".join(lines[i+1:]).strip()
            if block:
                blocks.append((info, block))
            break

        block = "\n".join(lines[i+1:j]).rstrip("\r\n")
        blocks.append((info, block))
        i = j + 1

    print("FOUND blocks:", [(repr(info), len(b)) for info,b in blocks])  # DEBUG

    def info_has_lang(s: str) -> bool:
        tokens = re.split(r"[^\w#+-]+", s.lower())
        return lang.lower() in tokens

    chosen = None
    for info, block in blocks:
        if info_has_lang(info):
            chosen = block
            break
    if not chosen and blocks:
        chosen = max(blocks, key=lambda ib: len(ib[1]))[1]

    if chosen:
        return chosen.strip()
    return parse_first_func(text, lang)




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
            print(f"stripped_line: {stripped_line}")
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