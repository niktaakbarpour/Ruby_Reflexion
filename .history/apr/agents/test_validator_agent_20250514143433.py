# multi_agent_apr/agents/test_validator_agent.py

class TestValidatorAgent:
    def __init__(self, model, gen):
        self.model = model
        self.gen = gen

    def validate_tests(self, item, tests, max_num_tests=5):
        problem_context = self._create_problem_context(item)
        return self.gen.validate_internal_tests(
            tests=tests,
            problem_context=problem_context,
            func=item["bug_source_code"],
            model=self.model,
            max_num_tests=max_num_tests
        )

    def _create_problem_context(self, item, include_buggy_code=False):
        outcome_descriptions = {
            "COMPILATION_ERROR": "The buggy code fails to compile or run due to a syntax error.",
            "RUNTIME_ERROR": "The code compiles successfully but encounters an error during execution.",
            "MEMORY_LIMIT_EXCEEDED": "The code uses more memory than allowed.",
            "TIME_LIMIT_EXCEEDED": "The code takes too long to run.",
            "WRONG_ANSWER": "The code runs but produces incorrect output.",
            "PASSED": "The buggy code passes all tests (may indicate insufficient tests).",
        }
        exec_outcome = item["bug_exec_outcome"]
        description = outcome_descriptions.get(exec_outcome, "Unknown execution outcome.")

        template = f"""Problem description: {item['description']}

Input format: {item['input_spec']}

Output format: {item['output_spec']}

A pre-run execution outcome of buggy source code: {exec_outcome} ({description})
"""
        if include_buggy_code:
            template = f"Buggy source code: {item['bug_source_code']}\n\n" + template

        return template
