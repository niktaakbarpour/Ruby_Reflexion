import re
from typing import Optional


def parse_code_block(string: str, lang: str) -> Optional[str]:
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    return parse_first_func(string, lang)

def parse_first_func(code: str, lang: str) -> Optional[str]:
    """
    Extract C++ code including pre-main lines (like #include, using namespace)
    and the entire main() function body.
    """
    print("parse_first_func!!!!!!!!!: ", code)
    if lang.lower() not in ("cpp", "c++"):
        raise AssertionError("Only C++ is supported for now")
    code_lines = code.split("\n")
    
    # Step 1: Collect pre-main lines (includes, using namespace, etc.)
    pre_main_lines = []
    main_start_idx = -1
    for i, line in enumerate(code_lines):
        stripped = line.strip()
        if stripped.startswith("#include") or stripped.startswith("using"):
            pre_main_lines.append(line)
        if re.match(r'int\s+main\s*\(.*\)\s*{', stripped):
            main_start_idx = i
            break
    
    if main_start_idx == -1:
        return None  # main not found
    
    # Step 2: Extract the main function body
    brace_count = 0
    main_lines = []
    in_main = False
    for i in range(main_start_idx, len(code_lines)):
        line = code_lines[i]
        stripped = line.strip()
        if not in_main:
            in_main = True
            brace_count += stripped.count("{") - stripped.count("}")
            main_lines.append(line)
            continue
        # Inside main
        brace_count += stripped.count("{")
        brace_count -= stripped.count("}")
        main_lines.append(line)
        if brace_count == 0:
            break

    # Step 3: Combine pre-main (excluding main declaration) + main function
    final_lines = pre_main_lines[:main_start_idx] + main_lines
    return "\n".join(final_lines)


def add_code_block(string: str, lang: str) -> str:
    return f"```{lang}\n{string}\n```"


if __name__ == "__main__":
    CODE = """
// Some random C++ code
dflkjfls
lsidjfklsdjf
lsdkfjklsjd
#include <iostream>
using namespace std;

int main() {
    int x = 1;
    if (x == 1) {
        return 1;
    } else {
        return 0;
    }
}

int another_func() {
    cout << "this should not be included" << endl;
    return 0;
}
"""
    print(parse_code_block(CODE, "cpp"))

    CODE = """
int main(vector<string> list1, vector<string> list2) {
    int total_chars_1 = 0;
    int total_chars_2 = 0;

    for (auto &s : list1) total_chars_1 += s.size();
    for (auto &s : list2) total_chars_2 += s.size();

    if (total_chars_1 <= total_chars_2)
        return 1;
    else
        return 0;
}
"""
    print(parse_code_block(CODE, "cpp"))