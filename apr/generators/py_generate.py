from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import (
    generic_generate_first_reflection,
    generic_generate_func_impl,
    generic_generate_internal_tests,
    generic_generate_self_reflection,
    generic_generate_scot_func_impl,
    generic_validate_internal_tests,
    generic_infer_specifications,
    generic_generate_debate_patch_evaluation,
)
from .prompt_constants import (
    RB_INFER_SPECIFICATIONS_FEW_SHOT,
    RB_INFER_SPECIFICATIONS_CHAT_INSTRUCTION,
    PY_SIMPLE_COMPLETION_INSTRUCTION,
    PY_REFLEXION_COMPLETION_INSTRUCTION,
    PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
    USE_PYTHON_CODEBLOCK_INSTRUCTION,
    PY_SIMPLE_CHAT_INSTRUCTION,
    PY_REFLEXION_CHAT_INSTRUCTION,
    PY_FIRST_REFLEXION_CHAT_INSTRUCTION,
    PY_REFLEXION_FEW_SHOT_ADD,
    PY_FIRST_REFLEXION_FEW_SHOT_ADD,
    PY_SELF_REFLECTION_CHAT_INSTRUCTION,
    PY_SELF_REFLECTION_FEW_SHOT,
    PY_FIRST_SELF_REFLECTION_FEW_SHOT,
    PY_TEST_GENERATION_FEW_SHOT,
    PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
    PY_TEST_GENERATION_CHAT_INSTRUCTION,
    RB_SCOT_CHAT_INSTRUCTION,
    RB_FIRST_SCOT_CHAT_INSTRUCTION,
    RB_REFLEXION_SCOT_FEW_SHOT_ADD,
    RB_FIRST_SCOT_FEW_SHOT,
    FIRST_REFLECTION_CHAT_INSTRUCTION,
    RB_TEST_GENERATION_EDGE_CHAT_INSTRUCTION,
    RB_TEST_GENERATION_EDGE_FEW_SHOT,
    RB_TEST_VALIDATION_IO_COT_FEW_SHOT,
    RB_TEST_VALIDATION_IO_COT_CHAT_INSTRUCTION,
    RB_TEST_GENERATION_IO_CHAT_INSTRUCTION,
    RB_REFLEXION_CHAT_INSTRUCTION_TEST_OMIT,
    RB_REFLEXION_FEW_SHOT_ADD_TEST_OMIT,
    RB_SELF_REFLECTION_CHAT_INSTRUCTION_TEST_OMIT,
    RB_SELF_REFLECTION_FEW_SHOT_TEST_OMIT,
    RB_FIRST_REFLEXION_CHAT_INSTRUCTION_FIRST_OMIT,
    RB_FIRST_REFLEXION_FEW_SHOT_ADD_FIRST_OMIT,
    RB_REFLEXION_CHAT_INSTRUCTION_SELF_OMIT,
    RB_REFLEXION_FEW_SHOT_ADD_SELF_OMIT,
    RB_DEBATE_PATCH_EVALUATION_CHAT_INSTRUCTION,
    RB_DEBATE_PATCH_EVALUATION_FEW_SHOT,
)

from .rb_parse import parse_code_block, add_code_block
from typing import Optional, List, Union
import re

class PyGenerator(Generator):
    def self_reflection(self,
                        func: str,
                        feedback: str,
                        model: ModelBase,
                        # inferred_specificaion:str
                    ) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            # inferred_specificaion=inferred_specificaion,
            self_reflection_chat_instruction_test_omit=RB_SELF_REFLECTION_CHAT_INSTRUCTION_TEST_OMIT,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "ruby"),
            self_reflection_few_shot_test_omit=RB_SELF_REFLECTION_FEW_SHOT_TEST_OMIT,
            self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT
        )

    def first_reflection(self,
                         problem_context: str,
                         func: str,
                         model: ModelBase,
                        #  inferred_specificaion: Optional[str]
                         ) -> str:
        return generic_generate_first_reflection(
            problem_context=problem_context,
            func=func,
            model=model,
            self_reflection_chat_instruction=FIRST_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "ruby"),
            self_reflection_few_shot=PY_FIRST_SELF_REFLECTION_FEW_SHOT,
            # inferred_specificaion=inferred_specificaion,
        )


    def func_impl(
        self,
        problem_context: str,
        # inferred_specificaion:str,
        model: ModelBase,
        strategy: str,
        num_comps: int,
        is_first_reflection: bool,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        reflections: Optional[str] = None,
        temperature: float = 1.0,
    ) -> Union[str, List[str]]:
        return generic_generate_func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            reflections=reflections,
            num_comps=num_comps,
            temperature=temperature,
            # inferred_specificaion=inferred_specificaion,
            reflexion_chat_instruction_test_omit=RB_REFLEXION_CHAT_INSTRUCTION_TEST_OMIT,
            reflexion_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
            first_reflexion_chat_instruction=PY_FIRST_REFLEXION_CHAT_INSTRUCTION,
            reflexion_few_shot=PY_REFLEXION_FEW_SHOT_ADD,
            first_reflexion_chat_instruction_first_omit=RB_FIRST_REFLEXION_CHAT_INSTRUCTION_FIRST_OMIT,
            reflexion_few_shot_test_omit=RB_REFLEXION_FEW_SHOT_ADD_TEST_OMIT,
            first_reflexion_few_shot=PY_FIRST_REFLEXION_FEW_SHOT_ADD,
            reflexion_few_shot_self_omit=RB_REFLEXION_FEW_SHOT_ADD_SELF_OMIT,
            first_reflexion_few_shot_first_omit=RB_FIRST_REFLEXION_FEW_SHOT_ADD_FIRST_OMIT,
            reflexion_chat_instruction_self_omit=RB_REFLEXION_CHAT_INSTRUCTION_SELF_OMIT,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "ruby"),
            add_code_block=lambda x: add_code_block(x, "ruby"),
        )

    def scot_func_impl(
        self,
        problem_context: str,
        model: ModelBase,
        strategy: str,
        num_comps: int,
        is_first_reflection: bool,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        reflections: Optional[str] = None,
        temperature: float = 1.0,
    ) -> Union[str, List[str]]:
        return generic_generate_scot_func_impl(
            problem_context=problem_context,
            model=model,
            strategy=strategy,
            is_first_reflection=is_first_reflection,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            reflections=reflections,
            num_comps=num_comps,
            temperature=temperature,
            reflexion_chat_instruction=RB_SCOT_CHAT_INSTRUCTION,
            first_reflexion_chat_instruction=RB_FIRST_SCOT_CHAT_INSTRUCTION,
            reflexion_few_shot=RB_REFLEXION_SCOT_FEW_SHOT_ADD,
            first_reflexion_few_shot=RB_FIRST_SCOT_FEW_SHOT,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
            simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "ruby"),
            add_code_block=lambda x: add_code_block(x, "ruby"),
        )

    def internal_tests(self,
                       samples: List[str],
                       problem_context: str,
                    #    inferred_specificaion:str,
                       func: str,
                       model: ModelBase,
                       max_num_tests: int = 7) -> List[str]:
        return generic_generate_internal_tests(
            problem_context=problem_context,
            func=func,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=RB_TEST_GENERATION_EDGE_FEW_SHOT,
            test_generation_chat_instruction=RB_TEST_GENERATION_EDGE_CHAT_INSTRUCTION,
            test_generation_completion_instruction=PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
            samples=samples,
            # inferred_specificaion=inferred_specificaion,
        )

    def debate_patch_evaluation(self, buggy_code: str, proposed_patch: str, problem_context: str, model: ModelBase) -> List[dict]:
        """Generate debate rounds for patch evaluation using two-agent debate strategy."""
        return generic_generate_debate_patch_evaluation(
            buggy_code=buggy_code,
            proposed_patch=proposed_patch,
            problem_context=problem_context,
            model=model,
            debate_chat_instruction=RB_DEBATE_PATCH_EVALUATION_CHAT_INSTRUCTION,
            debate_few_shot=RB_DEBATE_PATCH_EVALUATION_FEW_SHOT,
        )

    def validate_internal_tests(self, tests: List[str], problem_context: str, func: str, model: ModelBase, max_num_tests: int = 5) -> List[str]:
        return generic_validate_internal_tests(
            tests=tests,
            problem_context=problem_context,
            func=func,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=RB_TEST_VALIDATION_IO_COT_FEW_SHOT,
            test_generation_chat_instruction=RB_TEST_VALIDATION_IO_COT_CHAT_INSTRUCTION,
            test_generation_completion_instruction=PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
        )
    
    def infer_specification(self, problem_context: str, model: ModelBase) -> str:
        return generic_infer_specifications(
            problem_context=problem_context,
            model=model,
            infer_specifications_chat_instruction=RB_INFER_SPECIFICATIONS_FEW_SHOT,
            infer_specifications_few_shot=RB_INFER_SPECIFICATIONS_CHAT_INSTRUCTION,
        )
