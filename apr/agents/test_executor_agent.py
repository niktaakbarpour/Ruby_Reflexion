# multi_agent_apr/agents/test_executor_agent.py

class TestExecutorAgent:
    def __init__(self, executor):
        self.executor = executor

    def execute_and_evaluate(self, func_impl, tests, unittest_cases, timeout=10):
        result = self.executor.execute(func_impl, tests)
        is_passing = result.get("is_passing", False)
        feedback = result.get("feedback", "")

        if is_passing:
            is_passing = self.executor.evaluate(func_impl, unittest_cases, timeout=timeout)

        return {
            "is_passing": is_passing,
            "feedback": feedback
        }
