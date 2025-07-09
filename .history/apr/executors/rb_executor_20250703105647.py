import subprocess
import re
import sys
from typing import List, Tuple

class RbExecutor:
    def execute(self, func: str, tests: List[dict], timeout: int = 5) -> dict:
        """
        Executes a Ruby function against a list of test cases.

        Args:
            func (str): The Ruby function as a string.
            tests (List[dict]): List of test cases as dictionaries with "input" and "output".
            timeout (int): Timeout for each test execution (in seconds).

        Returns:
            dict: A dictionary containing:
                - is_passing (bool): Whether all tests passed.
                - feedback (str): Summary of test results.
                - state (List[bool]): List of pass/fail status for each test.
        """
        success_tests = []
        failed_tests = []
        is_passing = True
        
        for test in tests:
            try:
                result = subprocess.run(
                    ["/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby", "-e", func],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    input=test["input"],
                )

                actual_output = result.stdout.strip()
                
                # Normalize expected output, which could be a list or a single string
                expected_output = test["output"]
                if isinstance(expected_output, list):
                    expected_output = ''.join(expected_output).strip()  # If it's a list, join it into a single string
                else:
                    expected_output = expected_output.strip()  # If it's a string, just strip it

                # Normalize both actual and expected output for comparison
                if actual_output == expected_output:
                    success_tests.append(f"Input: {test['input']}, Expected: {expected_output}, Got: {actual_output}")
                else:
                    failed_tests.append(f"Input: {test['input']}, Expected: {expected_output}, Got: {actual_output}")
                    is_passing = False

            except subprocess.TimeoutExpired:
                failed_tests.append(f"Timeout for input: {test['input']}")
                is_passing = False
            except Exception as e:
                failed_tests.append(f"Error: {str(e)} for input: {test['input']}")
                is_passing = False
        
        feedback = ""
        if success_tests:
            feedback += "Tests passed:\n" + "\n".join(success_tests) + "\n\n"
        if failed_tests:
            feedback += "Tests failed:\n" + "\n".join(failed_tests)
        
        return {
            "is_passing": is_passing,
            "feedback": feedback,
            "state": [test in success_tests for test in tests],
        }


    def evaluate(self, func: str, test_cases: list, timeout: int = 5) -> Tuple[bool, List[dict]]:
        """
        Evaluates a Ruby function against multiple test cases and returns
        per-test results along with detailed debug printouts.

        Returns:
            - overall_pass (bool): True if all test cases pass.
            - detailed_results (List[dict]): List of test results with input, expected, actual, and pass/fail.
        """
        ruby_script_path = "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby"
        detailed_results = []
        overall_pass = True

        if test_cases and not isinstance(test_cases[0], dict):
            test_cases = [{"input": inp, "output": out} for inp, out in test_cases]

        for test in test_cases:
            test_input = test["input"]

            expected_output = test["output"]
            if isinstance(expected_output, list):
                expected_output = ''.join(expected_output).strip()
            else:
                expected_output = expected_output.strip()

            try:
                result = subprocess.run(
                    [ruby_script_path, "-e", func],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                actual_output = result.stdout.strip()

                # Debug prints
                print(f"actual_output: {actual_output}")
                print(f"expected_output: {expected_output}")

                passed = actual_output == expected_output
                if not passed:
                    print("Test failed")
                    overall_pass = False
                else:
                    print("Test passed")

                detailed_results.append({
                    "input": test_input.strip(),
                    "expected": expected_output,
                    "actual": actual_output,
                    "passed": passed
                })

            except subprocess.TimeoutExpired:
                print(f"Timeout for input: {test_input}")
                detailed_results.append({
                    "input": test_input.strip(),
                    "expected": expected_output,
                    "actual": "",
                    "passed": False,
                    "error": "Timeout"
                })
                overall_pass = False

            except Exception as e:
                print(f"Error executing Ruby code: {e}")
                detailed_results.append({
                    "input": test_input.strip(),
                    "expected": expected_output,
                    "actual": "",
                    "passed": False,
                    "error": str(e)
                })
                overall_pass = False

        if overall_pass:
            print("All tests passed")

        return overall_pass, detailed_results





# Example usage
if __name__ == "__main__":
  
    ruby_func = "def add(a, b); while true; x = 1; end; a + b; end"
    ruby_tests = ["assert_equal 3, add(1, 2)", "assert_equal 4, add(1, 2)"]

    # Create executor and run tests
    executor = RbExecutor()
    result = executor.execute(ruby_func, ruby_tests, timeout=2)
    print(result["feedback"])