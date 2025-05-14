class ProgrammerAgent:
    def __init__(self, model, strategy, gen, alt_gen=None):
        self.model = model
        self.strategy = strategy
        self.gen = gen
        self.alt_gen = alt_gen if alt_gen is not None else gen

    def generate(self, item, prev_impl, reflection=None, feedback=None, is_first=False, prompting="scot"):
        problem_context = self._create_problem_context(item)

        if prompting == "scot":
            return self.gen.scot_func_impl(
                problem_context=problem_context,
                model=self.model,
                strategy=self.strategy,
                is_first_reflection=is_first,
                prev_func_impl=prev_impl,
                self_reflection=reflection,
                feedback=feedback
            )
        else:
            return self.alt_gen.func_impl(
                problem_context=problem_context,
                model=self.model,
                strategy=self.strategy,
                is_first_reflection=is_first,
                prev_func_impl=prev_impl,
                self_reflection=reflection,
                feedback=feedback
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
