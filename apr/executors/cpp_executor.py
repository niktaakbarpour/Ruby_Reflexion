import subprocess
import re
import sys
from typing import List, Tuple, Dict
import signal
import tempfile
import os

class CppExecutor:
    def _classify_failure(self, result, expected_output, actual_output: str, timeout_expired: bool) -> Tuple[str, str]:
        """
        Classify failure for C++ runs (and optionally for compile_result if passed here).
        Returns (verdict, info).
        """

        if timeout_expired:
            return "TIME LIMIT EXCEEDED", "Process exceeded wall-time limit."

        if result is None:
            return "RUNTIME ERROR", "No result returned."

        stderr = (result.stderr or "").strip()
        rc = result.returncode

        # --- If this was a compile step and compiler emitted errors, report COMPILATION ERROR ---
        # (In typical flow we separate compile_result, but keep this check if compile_result is passed)
        if stderr:
            lower = stderr.lower()
            if ("error:" in stderr) or ("undefined reference" in lower) or ("ld:" in lower) or ("collect2" in lower):
                # limit message length so feedback stays readable
                info = stderr if len(stderr) < 2000 else stderr[:2000] + "\n... (truncated)"
                return "COMPILATION ERROR", info

        # --- Check termination by signal ---
        sig = None
        if isinstance(rc, int):
            if rc < 0:
                sig = -rc
            elif rc >= 128:
                # shell-style code 128 + sig
                sig = rc - 128

        if sig is not None:
            # map some common signals to friendly verdicts
            if sig == 9:   # SIGKILL (often OOM killer)
                return "MEMORY LIMIT EXCEEDED", f"Terminated by SIGKILL (signal {sig}). stderr: {stderr}"
            if sig == 24:  # SIGXCPU (CPU time limit exceeded)
                return "TIME LIMIT EXCEEDED", f"Terminated by SIGXCPU (signal {sig}). stderr: {stderr}"
            if sig == 11:  # SIGSEGV
                return "RUNTIME ERROR", f"Segmentation fault (signal {sig}). stderr: {stderr}"
            if sig == 7:   # SIGBUS
                return "RUNTIME ERROR", f"Bus error (signal {sig}). stderr: {stderr}"
            if sig == 8:   # SIGFPE
                return "RUNTIME ERROR", f"Floating point exception (signal {sig}). stderr: {stderr}"
            if sig == 6:   # SIGABRT
                return "RUNTIME ERROR", f"Abort called (signal {sig}). stderr: {stderr}"
            # generic signal case
            return "RUNTIME ERROR", f"Terminated by signal {sig}. stderr: {stderr}"

        # --- Non-zero exit code with stderr -> runtime error (e.g., std::terminate, exceptions, sanitizer output) ---
        if rc != 0 and stderr:
            info = stderr if len(stderr) < 2000 else stderr[:2000] + "\n... (truncated)"
            return "RUNTIME ERROR", info

        # --- Normalize expected_output (support string or list) ---
        if isinstance(expected_output, (list, tuple)):
            expected_list = [e.strip() for e in expected_output]
        else:
            expected_list = [expected_output.strip()]

        # normalize actual output
        actual_norm = actual_output.strip()

        # --- Wrong answer check (allow any of expected_list) ---
        if actual_norm not in expected_list:
            return "WRONG ANSWER", f"Expected one of: {expected_list}. Got: {actual_norm}"

        # success
        return "SUCCESS", ""

    def execute(self, source_code: str, tests: List[Dict], timeout: int = 5) -> dict:
        """
        Executes C++ code against test cases:
        - Compiles the source code first.
        - For each test: runs the executable with test input.
        
        Args:
            source_code (str): Full C++ source code as string.
            tests (List[Dict]): List of {"input": str, "output": str}.
            timeout (int): Timeout for each test in seconds.

        Returns:
            dict with keys: is_passing, feedback, state (list bool), results (list dict)
        """

        success_tests = []
        failed_tests = []
        state = []
        detailed_results = []
        is_passing = True

        # Step 1: Write source code to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = os.path.join(tmpdir, "prog.cpp")
            exe_file = os.path.join(tmpdir, "prog.out")

            with open(source_file, "w") as f:
                f.write(source_code)

            # Step 2: Compile the source code
            compile_cmd = ["g++", "-std=c++17", source_file, "-o", exe_file]
            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return {
                    "is_passing": False,
                    "feedback": "COMPILATION TIME LIMIT EXCEEDED",
                    "state": [False] * len(tests),
                    "results": []
                }

            # Check compilation error
            if compile_result.returncode != 0:
                # Compilation failed, report error
                feedback = "COMPILATION ERROR:\n" + compile_result.stderr.strip()
                return {
                    "is_passing": False,
                    "feedback": feedback,
                    "state": [False] * len(tests),
                    "results": []
                }

            # Step 3: Run tests one by one
            for test in tests:
                test_input = test["input"]
                expected_output = test["output"].strip()

                try:
                    run_result = subprocess.run(
                        [exe_file],
                        input=test_input,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    actual_output = run_result.stdout.strip()

                    # Use _classify_failure to decide verdict
                    verdict, info = self._classify_failure(run_result, expected_output, actual_output, False)

                except subprocess.TimeoutExpired:
                    verdict, info = self._classify_failure(None, expected_output, "", True)
                    actual_output = ""
                    is_passing = False
                    state.append(False)
                    failed_tests.append(f"{verdict} for input: {test_input}. Details: {info}")
                    detailed_results.append({
                        "input": test_input,
                        "expected": expected_output,
                        "actual": "",
                        "verdict": verdict,
                        "info": info,
                        "passed": False
                    })
                    continue

                except Exception as e:
                    verdict, info = "RUNTIME ERROR", str(e)
                    actual_output = ""
                    is_passing = False
                    state.append(False)
                    failed_tests.append(f"{verdict} for input: {test_input}. Details: {info}")
                    detailed_results.append({
                        "input": test_input,
                        "expected": expected_output,
                        "actual": "",
                        "verdict": verdict,
                        "info": info,
                        "passed": False
                    })
                    continue

                if verdict == "SUCCESS":
                    success_tests.append(f"Input: {test_input}, Output: {actual_output}, Verdict: {verdict}")
                    state.append(True)
                else:
                    failed_tests.append(f"{verdict} for input: {test_input}. Details: {info}")
                    state.append(False)
                    is_passing = False

                detailed_results.append({
                    "input": test_input,
                    "expected": expected_output,
                    "actual": actual_output,
                    "verdict": verdict,
                    "info": info,
                    "passed": verdict == "SUCCESS"
                })

            # Combine feedback
            feedback = ""
            if success_tests:
                feedback += "Tests passed:\n" + "\n".join(success_tests) + "\n\n"
            if failed_tests:
                feedback += "Tests failed:\n"
                for detail in detailed_results:
                    if not detail["passed"]:
                        fail_msg = f"{detail['verdict']} for input: {detail['input']}. Details: {detail['info']}"
                        feedback += f"{fail_msg} | Verdict: {detail['verdict']} | Info: {detail['info']}\n"

            return {
                "is_passing": is_passing,
                "feedback": feedback,
                "state": state,
                "results": detailed_results
            }

    def evaluate(self, source_code: str, test_cases: list, timeout: int = 5) -> Tuple[bool, List[dict]]:
        """
        Evaluate C++ code against multiple test cases and return detailed structured results for each test.
        
        Args:
            source_code: Complete C++ source code as a string (will be compiled).
            test_cases: List of test cases, each a dict with "input" and "output".
            timeout: Time limit for each compilation and execution in seconds.
        
        Returns:
            overall_pass: Boolean indicating if all test cases passed.
            detailed_results: List of dicts with info about each test's input, expected, actual, verdict, etc.
        """

        detailed_results = []
        overall_pass = True

        # Convert simple test cases (e.g., list of tuples) to dict format if necessary
        if test_cases and not isinstance(test_cases[0], dict):
            test_cases = [{"input": inp, "output": out} for inp, out in test_cases]

        # Create a temporary directory to hold source and executable files
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = os.path.join(tmpdir, "prog.cpp")
            exe_file = os.path.join(tmpdir, "prog.out")

            # Write the C++ source code to a file
            with open(source_file, "w") as f:
                f.write(source_code)

            # Compile the C++ source code
            compile_cmd = ["g++", "-std=c++17", source_file, "-o", exe_file]
            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                # Compilation timed out
                for test in test_cases:
                    detailed_results.append({
                        "input": test["input"],
                        "expected": test["output"],
                        "actual": "",
                        "verdict": "COMPILATION TIME LIMIT EXCEEDED",
                        "info": "",
                        "passed": False
                    })
                return False, detailed_results

            # Check for compilation errors
            if compile_result.returncode != 0:
                info = compile_result.stderr.strip()
                for test in test_cases:
                    detailed_results.append({
                        "input": test["input"],
                        "expected": test["output"],
                        "actual": "",
                        "verdict": "COMPILATION ERROR",
                        "info": info,
                        "passed": False
                    })
                return False, detailed_results

            # Run each test case against the compiled executable
            for test in test_cases:
                test_input = test["input"].strip()
                expected_output = test["output"]
                if isinstance(expected_output, list):
                    expected_outputs = [o.strip() for o in expected_output]
                else:
                    expected_outputs = [expected_output.strip()]

                try:
                    run_result = subprocess.run(
                        [exe_file],
                        input=test_input,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    actual_output = run_result.stdout.strip()

                    # Check if actual output matches any expected output
                    if actual_output in expected_outputs:
                        verdict, info = "SUCCESS", "Output matched"
                    else:
                        verdict, info = self._classify_failure(run_result, expected_outputs, actual_output, False)

                except subprocess.TimeoutExpired:
                    # Execution timed out
                    verdict, info = self._classify_failure(None, expected_outputs, "", True)
                    actual_output = ""

                except Exception as e:
                    # Runtime error during execution
                    verdict, info = "RUNTIME ERROR", str(e)
                    actual_output = ""

                passed = (verdict == "SUCCESS")
                if not passed:
                    overall_pass = False

                # Record detailed result for this test
                detailed_results.append({
                    "input": test_input,
                    "expected": expected_outputs,
                    "actual": actual_output,
                    "verdict": verdict,
                    "info": info,
                    "passed": passed
                })

        return overall_pass, detailed_results



if __name__ == "__main__":
    # Faulty C++ program (infinite loop to trigger TIME LIMIT EXCEEDED)
    cpp_code = """
    #include <iostream>
    using namespace std;
    int add(int a, int b) {
        return a + b;
    }
    int main() {
        int x, y;
        cin >> x >> y;
        cout << add(x, y) << endl;
        return 0;
    }
    """
    # Proper input/output test cases
    cpp_tests = [
        {"input": "1 2", "output": "3"},
        {"input": "2 2", "output": "4"},
    ]

    executor = CppExecutor()
    result = executor.evaluate(cpp_code, cpp_tests, timeout=2)

    # Print summarized feedback
    for r in result[1]:
        print(r)

    # If you want a summary:
    print("All tests passed:", result[0])

