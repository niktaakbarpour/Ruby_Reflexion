from vllm import LLM, CompletionConfig, SamplingParams
from typing import List, Union, Optional, Literal
import dataclasses
import os
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps)


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")


class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0, num_comps: int = 1) -> Union[List[str], str]:
        import torch
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.tolist()  # Convert tensor to list of token IDs

        # Decode outputs
        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        # Ensure `outs` is a list of strings
        assert isinstance(outs, list), f"Expected outs to be a list, got {type(outs)}"
        for i, out in enumerate(outs):
            assert isinstance(out, str), f"Expected out to be a string, got {type(out)}"

        # If you need additional processing, call extract_output only if necessary
        # Assuming extract_output applies further processing on decoded strings:
        outs = [self.extract_output(out) for out in outs]

        # Return the decoded output(s)
        if len(outs) == 1:
            return outs[0]  # Return a single string
        else:
            return outs  # Return a list of strings

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)



    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]

        return out

class DeepSeekCoder(HFModelBase):
    import torch
    def __init__(self, model_path=None):
        from typing import List, Union
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        """
        Initialize the DeepseekCoder model.

        :param model_path: local path to the model if you have downloaded it locally.
        """

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
            torch_dtype="auto",
            quantization_config=nf4_config,
            trust_remote_code=True
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
            trust_remote_code=True
        )

        super().__init__("deepseek-ai/deepseek-coder-6.7b-instruct", model, tokenizer)
class DeepSeekR1(ModelBase):
    def __init__(
        self,
        model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,   # 7B fits on 1×A100-40GB; set 2 if you allocate 2 GPUs
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ):
        self.name = "deepseek-r1-distill-qwen-7b"
        self.is_chat = True

        base_path = (
            model_path
            or os.environ.get("MODEL_PATH")
            or os.environ.get("DEEPSEEK_MODEL_PATH")
            or os.environ.get("SLURM_TMPDIR")
        )
        if base_path is None:
            raise ValueError("Provide model_path or set MODEL_PATH/DEEPSEEK_MODEL_PATH/SLURM_TMPDIR.")

        actual_model_path = self.find_model_directory(base_path)

        self.llm = LLM(
            model=actual_model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
        )

    @staticmethod
    def find_model_directory(model_path: str) -> str:
        p = Path(model_path)
        if (p / "config.json").exists():
            return str(p)
        snapshots_dir = p / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if snapshot_dirs:
                actual_path = snapshot_dirs[0]
                if (actual_path / "config.json").exists():
                    return str(actual_path)
        raise ValueError(
            f"Could not find valid model directory in {model_path}. "
            f"Expected either config.json directly or in snapshots subdirectory."
        )

    def _default_stop(self) -> List[str]:
        return ["<|User|>", "<|Assistant|>", "\n\nUser:", "\n\nAssistant:"]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.6,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=stop_strs or self._default_stop(),
            n=num_comps,
        )
        results = self.llm.generate([prompt], sampling_params)
        texts = [o.text for o in results[0].outputs]
        return texts[0] if num_comps == 1 else texts

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        temperature: float = 0.6,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        assert messages and messages[-1].role == "user", "Last message must be from user"

        parts: List[str] = []
        for m in messages:
            if m.role == "user":
                parts.append(f"<|User|>{m.content}")
            elif m.role == "assistant":
                parts.append(f"<|Assistant|>{m.content}")
            elif m.role == "system":
                parts.append(f"{m.content}\n")

        prompt = "".join(parts) + "<|Assistant|>"
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            stop_strs=self._default_stop(),
            temperature=temperature,
            num_comps=num_comps,
        )

class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b", model_path = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path is not None else f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path if model_path is not None else f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out