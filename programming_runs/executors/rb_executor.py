import subprocess
import re
import sys
from typing import List, Tuple

class RbExecutor:
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> dict:
        """
        Executes a Ruby function against a list of test cases.
        
        Args:
            func (str): The Ruby function as a string.
            tests (List[str]): List of test cases as strings.
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


        func_name = func.split()[1].split('(')[0]  # Extract function name
        ruby_code = f"require 'test/unit'\n\n"
        ruby_code += func + "\n\n"

        ruby_code += f"class TestHumanEval < Test::Unit::TestCase\n"
        ruby_code += f"  def test_{func_name}\n"
        ruby_code += f"    candidate = method(:{func_name})\n"
        for test in tests:
            # Replace only function calls in test cases
            modified_test = re.sub(rf"(?<!\w){func_name}\s*\(", "candidate.call(", test)
            ruby_code += f"    {modified_test}\n"
        ruby_code += "  end\n"
        ruby_code += "end\n"


        print(f"ruby_code4: {ruby_code}")

        try:
            result = subprocess.run(
                ["/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby", "-e", ruby_code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            print(f"RESULTSSSSS: {result}")

            # Parse results
            if result.returncode == 0:
                success_tests = tests
            else:
                failed_tests.append(f"Test suite failed # output: {result.stdout.strip()} {result.stderr.strip()}")
                is_passing = False

        except subprocess.TimeoutExpired:
            failed_tests.append("Test suite timed out")
            is_passing = False
        except Exception as e:
            failed_tests.append(f"Error: {str(e)}")
            is_passing = False

        # Generate feedback
        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\n"
        feedback += "Tests failed:\n" + "\n".join(failed_tests)

        return {
            "is_passing": is_passing,
            "feedback": feedback,
            "state": [test in success_tests for test in tests],
        }

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates a Ruby function against a single test case.
        
        Args:
            name (str): Name of the function.
            func (str): The Ruby function as a string.
            test (str): The test case as a string.
            timeout (int): Timeout for execution (in seconds).
        
        Returns:
            bool: True if the test passes, False otherwise.
        """
        ruby_code = f"{func}\n{test}"
        try:
            result = subprocess.run(
                ["ruby", "-e", ruby_code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0 and "true" in result.stdout.lower()
        except:
            return False


# Example usage
if __name__ == "__main__":
  
    ruby_func = "def add(a, b); while true; x = 1; end; a + b; end"
    ruby_tests = ["assert_equal 3, add(1, 2)", "assert_equal 4, add(1, 2)"]

    # Create executor and run tests
    executor = RbExecutor()
    result = executor.execute(ruby_func, ruby_tests, timeout=2)
    print(result["feedback"])