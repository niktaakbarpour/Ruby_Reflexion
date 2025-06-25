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


    def evaluate(self, func: str, test_cases: list, timeout: int = 5) -> bool:
        """
        Evaluates a Ruby function against multiple test cases.

        Args:
            func (str): The Ruby function as a string.
            test_cases (list): A list of dictionaries containing "input" and "output".
            timeout (int): Timeout for execution (in seconds).

        Returns:
            bool: True if all test cases pass, False otherwise.
        """
        ruby_script_path = "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby"

        for test in test_cases:
            test_input = test["input"]
            
            # Normalize expected output, which could be a list or a single string
            expected_output = test["output"]
            if isinstance(expected_output, list):
                expected_output = ''.join(expected_output).strip()  # If it's a list, join it into a single string
            else:
                expected_output = expected_output.strip()  # If it's a string, just strip it

            try:
                result = subprocess.run(
                    [ruby_script_path, "-e", func],  # Run the Ruby function
                    input=test_input,  # Pass input via stdin
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                actual_output = result.stdout.strip()  # Normalize actual output

                # Debugging output
                print(f"actual_output: {actual_output}")
                print(f"expected_output: {expected_output}")

                if actual_output != expected_output:
                    print("Test failed")
                    return False  # Fail if any test case doesn't match

            except subprocess.TimeoutExpired:
                print(f"Timeout for input: {test_input}")
                return False  # Fail if the process times out

            except Exception as e:
                print(f"Error executing Ruby code: {e}")
                return False  # Fail on any other exception
        
        print("All tests passed")
        return True  # Pass only if all test cases succeed




# Example usage
if __name__ == "__main__":
  
    ruby_func = "def add(a, b); while true; x = 1; end; a + b; end"
    ruby_tests = ["assert_equal 3, add(1, 2)", "assert_equal 4, add(1, 2)"]

    # Create executor and run tests
    executor = RbExecutor()
    result = executor.execute(ruby_func, ruby_tests, timeout=2)
    print(result["feedback"])