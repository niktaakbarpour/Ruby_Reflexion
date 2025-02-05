import subprocess
import ast
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

        # Extract the function name dynamically
        func_name = func.split()[1].split('(')[0]

        print(f"func_name: {func_name}")

        # Build the Ruby test script dynamically
        ruby_code = f"require 'test/unit'\n\n"
        ruby_code += f"def {func_name}(...)\n  # function implementation\nend\n\n"
        ruby_code += f"class TestHumanEval < Test::Unit::TestCase\n"
        ruby_code += f"  def test_{func_name}\n"
        ruby_code += f"    candidate = method(:{func_name})\n"
        
        # Add all the test cases inside the test method
        for test in tests:
            ruby_code += f"    {test}\n"

        ruby_code += "  end\n"
        ruby_code += "end\n"

        print(f"ruby_code4: {ruby_code}")

        try:
            # Execute the Ruby code using `ruby` command
            result = subprocess.run(
                ["/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/ruby/2.7.1/bin/ruby", "-e", ruby_code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            print(f"RESULTSSSSS: {result}")

            if result.stdout:
                print(f"stdout: {result.stdout}")
            if result.stderr:
                print(f"stderr: {result.stderr}")


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


        # for test in tests:
        #     # Combine the function and test case
        #     ruby_code = f"{func}\n{test}"
        #     print(f"ruby_code: {ruby_code}")
            
        #     try:
        #         # Execute the Ruby code using the `ruby` command
        #         result = subprocess.run(
        #             ["ruby", "-e", ruby_code],
        #             capture_output=True,
        #             text=True,
        #             timeout=timeout,
        #         )
        #         print(f"result: {result}")
        #         # Check if the test passed
        #         if result.returncode == 0 and "true" in result.stdout.lower():
        #             success_tests.append(test)
        #         else:
        #             failed_tests.append(f"{test} # output: {result.stdout.strip()}")
        #             is_passing = False
        #     except subprocess.TimeoutExpired:
        #         failed_tests.append(f"{test} # output: TIMEOUT")
        #         is_passing = False
        #     except Exception as e:
        #         failed_tests.append(f"{test} # output: {str(e)}")
        #         is_passing = False

        # # Generate feedback
        # feedback = "Tests passed:\n"
        # feedback += "\n".join(success_tests) + "\n\n"
        # feedback += "Tests failed:\n"
        # feedback += "\n".join(failed_tests)

        # # Return results
        # return {
        #     "is_passing": is_passing,
        #     "feedback": feedback,
        #     "state": [test in success_tests for test in tests],
        # }

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
    # Ruby function to test
    ruby_func = """
def find_closest_elements(numbers)
  if numbers.length < 2
    raise ArgumentError, "The input list must contain at least two elements."
  end

  closest_pair = nil
  min_diff = Float::INFINITY

  numbers.each_with_index do |num1, i|
    numbers[i+1..-1].each do |num2|
      diff = (num1 - num2).abs
      if diff < min_diff
        min_diff = diff
        closest_pair = [num1, num2].sort
      end
    end
  end

  closest_pair
end
"""

    # Test cases
    ruby_tests = [
        'puts find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == [2.0, 2.2]',
        'puts find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == [2.0, 2.0]',
        'puts find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0]) == [1.0, 2.0]',
        'puts find_closest_elements([5.0, 4.0, 3.0, 2.0, 1.0]) == [1.0, 2.0]',
        'puts find_closest_elements([1.0, 3.0, 2.0, 4.0, 5.0]) == [1.0, 2.0]',
    ]

    # Create executor and run tests
    executor = RbExecutor()
    result = executor.execute(ruby_func, ruby_tests, timeout=2)
    print(result["feedback"])