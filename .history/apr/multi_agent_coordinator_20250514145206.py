# multi_agent_apr/multi_agent_coordinator.py

class MultiAgentCoordinator:
    def __init__(self, programmer, test_designer, test_validator, test_executor, feedback_agent):
        self.programmer = programmer
        self.test_designer = test_designer
        self.test_validator = test_validator
        self.test_executor = test_executor
        self.feedback_agent = feedback_agent

    def run(self, item, max_iters=3, pass_at_k=1, prompting="scot"):
        reflections = []
        implementations = []
        test_feedback = []
        is_first = True
        is_solved = False
        num_success = 0
        cur_pass = 0

        # Initial setup
        reflection = self.feedback_agent.first_reflection(item)
        reflections.append(reflection)

        test_cases = self.test_designer.generate_tests(item)
        validated_tests = self.test_validator.validate_tests(item, test_cases)
        formatted_tests = [{"input": i, "output": o} for i, o in validated_tests]

        cur_impl = item["bug_source_code"]

        while cur_pass < pass_at_k and not is_solved:
            impl = self.programmer.generate(
                item=item,
                prev_impl=cur_impl,
                reflection=reflection,
                feedback=test_feedback[-1] if test_feedback else None,
                is_first=is_first,
                prompting=prompting
            )
            implementations.append(impl)

            result = self.test_executor.execute_and_evaluate(
                func_impl=impl,
                tests=formatted_tests,
                unittest_cases=item["unittest_cases"]
            )

            test_feedback.append(result["feedback"])
            if result["is_passing"]:
                is_solved = True
                num_success += 1
                break

            # Iterative reflection loop
            for iter_idx in range(1, max_iters):
                reflection = self.feedback_agent.self_reflect(impl, result["feedback"])
                reflections.append(reflection)

                impl = self.programmer.generate(
                    item=item,
                    prev_impl=impl,
                    reflection=reflection,
                    feedback=result["feedback"],
                    is_first=False,
                    prompting=prompting
                )
                implementations.append(impl)

                result = self.test_executor.execute_and_evaluate(
                    func_impl=impl,
                    tests=formatted_tests,
                    unittest_cases=item["unittest_cases"]
                )
                test_feedback.append(result["feedback"])

                if result["is_passing"]:
                    is_solved = True
                    num_success += 1
                    break

            cur_pass += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = implementations[-1] if implementations else cur_impl

        return item, num_success
