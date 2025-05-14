# multi_agent_apr/run_reflexion_multi_agent.py

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from agents.programmer_agent import ProgrammerAgent
from agents.test_designer_agent import TestDesignerAgent
from agents.test_validator_agent import TestValidatorAgent
from agents.test_executor_agent import TestExecutorAgent
from agents.feedback_integration_agent import FeedbackIntegrationAgent
from multi_agent_coordinator import MultiAgentCoordinator


def run_reflexion_multi_agent(
    dataset,
    model_name,
    language,
    max_iters,
    pass_at_k,
    log_path,
    verbose,
    is_leetcode=False,
    model_path=None
):
    prompting = "scot"
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name, model_path)

    # Instantiate agents
    programmer = ProgrammerAgent(model=model, strategy="reflexion", gen=gen)
    test_designer = TestDesignerAgent(model=model, gen=gen)
    test_validator = TestValidatorAgent(model=model, gen=gen)
    test_executor = TestExecutorAgent(executor=exe)
    feedback_agent = FeedbackIntegrationAgent(model=model, gen=gen)

    # Coordinator
    coordinator = MultiAgentCoordinator(
        programmer=programmer,
        test_designer=test_designer,
        test_validator=test_validator,
        test_executor=test_executor,
        feedback_agent=feedback_agent
    )

    print_v = make_printv(verbose)
    num_items = len(dataset)
    total_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path):
        try:
            updated_item, num_success = coordinator.run(
                item=item,
                max_iters=max_iters,
                pass_at_k=pass_at_k,
                prompting=prompting
            )
            write_jsonl(log_path, [updated_item], append=True)
            total_success += num_success
            print_v(f"completed {i+1}/{num_items}: acc = {round(total_success / (i + 1), 2)}")
        except Exception as e:
            print(f"Error processing item {i}: {e}. Continuing with next item.")
            continue
