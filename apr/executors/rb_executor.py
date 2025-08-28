import subprocess
import re
import sys
import os
from typing import List, Tuple

class RbExecutor:
    def _classify_failure(self, result, expected_output: str, actual_output: str, timeout_expired: bool) -> Tuple[str, str]:

        if timeout_expired:
            return "TIME LIMIT EXCEEDED", ""

        if result is None:
            return "RUNTIME ERROR", "No result returned"

        stderr = result.stderr.strip()
        rc = result.returncode

        # COMPILATION ERROR
        if "SyntaxError" in stderr or "unexpected" in stderr:
            return "COMPILATION ERROR", stderr

        # MEMORY LIMIT EXCEEDED
        if "killed" in stderr.lower() or rc == 137:
            return "MEMORY LIMIT EXCEEDED", stderr

        # RUNTIME ERROR
        if rc != 0 and stderr:
            return "RUNTIME ERROR", stderr

        # WRONG ANSWER
        if actual_output.strip() != expected_output.strip():
            return "WRONG ANSWER", f"Expected: {expected_output.strip()}, Got: {actual_output.strip()}"

        # SUCCESS
        return "SUCCESS", ""



    def execute(self, func: str, tests: List[dict], timeout: int = 5) -> dict:
        """
        Executes a Ruby function against a list of test cases.

        Args:
            func (str): The Ruby function as a string.
            tests (List[dict]): List of test cases with "input" and "output".
            timeout (int): Timeout for each test execution (in seconds).

        Returns:
            dict: {
                - is_passing (bool): All tests passed?
                - feedback (str): Summary log
                - state (List[bool]): Pass/fail for each test
                - results (List[dict]): Per-test structured outcome
            }
        """
        ruby_path = "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby"
        success_tests = []
        failed_tests = []
        state = []
        detailed_results = []
        is_passing = True

        for test in tests:
            test_input = test["input"].strip()
            expected_output = test["output"]
            if isinstance(expected_output, list):
                expected_output = ''.join(expected_output).strip()
            else:
                expected_output = expected_output.strip()

            try:
                result = subprocess.run(
                    [ruby_path, "-e", func],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                actual_output = result.stdout.strip()
                verdict, info = self._classify_failure(result, expected_output, actual_output, False)

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

            # If no exception occurred, proceed with verdict check
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

        # Combine feedback string
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





    def evaluate(self, func: str, test_cases: list, timeout: int = 5) -> Tuple[bool, List[dict]]:
        """
        Evaluates a Ruby function against multiple test cases and returns
        per-test results with structured failure reasons.

        Returns:
            - overall_pass (bool): True if all test cases pass.
            - detailed_results (List[dict]): Each dict contains:
                - input, expected, actual, verdict, info, passed
        """
        # Avoid tokenizers fork warnings bleeding into logs
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        ruby_path = "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby"
        detailed_results: List[dict] = []
        overall_pass = True

        # Accept [(inp, out), ...] and convert to [{"input":..., "output":...}, ...]
        if test_cases and not isinstance(test_cases[0], dict):
            test_cases = [{"input": inp, "output": out} for inp, out in test_cases]
            print(f"Ruby executor test cases: {test_cases}")

        def _coerce_input(x) -> str:
            """Return a newline-terminated string to feed to Ruby."""
            if isinstance(x, str):
                s = x.strip()
            elif isinstance(x, (list, tuple)):
                # Join multi-line / multi-arg inputs by newline
                s = "\n".join(str(v).strip() for v in x)
            else:
                s = str(x).strip()
            # Ensure trailing newline so Ruby's gets/STDIN works as expected
            if not s.endswith("\n"):
                s += "\n"
            return s

        def _coerce_expected(x) -> List[str]:
            if isinstance(x, (list, tuple)):
                return [str(o).strip() for o in x]
            return [str(x).strip()]

        for test in test_cases:
            print(f"Ruby executor test: {test}")

            # Normalize input/expected early so exception handlers can use them
            test_input = _coerce_input(test.get("input", ""))
            expected_outputs = _coerce_expected(test.get("output", ""))

            print(f"Ruby executor test_input: {repr(test_input)}")
            print(f"Ruby executor expected_outputs: {expected_outputs}")

            try:
                result = subprocess.run(
                    [ruby_path, "-e", func],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                print(f"Ruby executor result: {result}")

                actual_output = (result.stdout or "").strip()
                print(f"Ruby executor actual_output: {actual_output}")

                if result.returncode != 0:
                    # Ruby crashed or raised => runtime error
                    verdict, info = "RUNTIME ERROR", (result.stderr or "").strip()
                elif actual_output in expected_outputs:
                    verdict, info = "SUCCESS", "Output matched"
                else:
                    # Ruby ran, but output doesn't match => wrong answer
                    verdict, info = "WRONG ANSWER", f"Expected one of {expected_outputs}, got '{actual_output}'"

            except subprocess.TimeoutExpired:
                verdict, info = "TIMEOUT", f"Exceeded {timeout}s"
                actual_output = ""
            except Exception as e:
                # Python-side error (e.g., bad input shape) should not be shown as runtime error for the submission
                verdict, info = "HARNESS ERROR", str(e)
                actual_output = ""

            passed = verdict == "SUCCESS"
            if not passed:
                overall_pass = False

            detailed_results.append({
                "input": test_input.rstrip("\n"),
                "expected": expected_outputs,
                "actual": actual_output,
                "verdict": verdict,
                "info": info,
                "passed": passed,
            })

        return overall_pass, detailed_results





if __name__ == "__main__":
    # Faulty Ruby function (infinite loop to trigger TIME LIMIT EXCEEDED)
    ruby_func = "def add(a, b); while true; x = 1; end; a + b; end"

    # Proper input/output test cases
    ruby_tests = [
        {"input": "puts add(1, 2)", "output": "3"},
        {"input": "puts add(2, 2)", "output": "4"},
    ]

    executor = RbExecutor()
    result = executor.execute(ruby_func, ruby_tests, timeout=2)
    print(result["feedback"])

    # Optional: inspect structured results
    for r in result["results"]:
        print(r)