from typing import List, Union, Optional, Literal
import dataclasses

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

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
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

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
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

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

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
        from accelerate import init_empty_weights, infer_auto_device_map
        import torch
        
        """
        Initialize the DeepseekCoder model.

        :param model_path: local path to the model if you have downloaded it locally.
        """
        # Print CUDA availability
        print(torch.cuda.is_available())

        print("I'm Here")

        # Initialize the model with empty weights to avoid memory overload
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
        #         trust_remote_code=True
        #     )

        print("I'm Here 1")

        # Infer a device map to offload layers as needed
        # device_map = infer_auto_device_map(model, no_split_module_classes=["CausalLM"])

        print("I'm Here 2")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )

        print("I'm Here 3")

        model = AutoModelForCausalLM.from_pretrained(
            model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
            # device_map=device_map,
            torch_dtype="auto",
            quantization_config=nf4_config,
            trust_remote_code=True
        )

        # Load the model using the device map and float32 precision
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
        #     device_map=device_map,
        #     torch_dtype=torch.float16,
        #     load_in_8bit=True,
        #     trust_remote_code=True
        # )

        

        # Apply dynamic quantization to reduce memory usage
        # model = torch.quantization.quantize_dynamic(
        #     model,
        #     {torch.nn.Linear},  # Apply quantization to Linear layers
        #     dtype=torch.qint8    # Quantize weights to int8
        # )

        print("I'm Here 4")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
            trust_remote_code=True
        )

        print("I'm Here 5")

        # Print the model's device map for debugging
        # print("Device map:", device_map)

        # Initialize the base class
        super().__init__("deepseek-ai/deepseek-coder-6.7b-instruct", model, tokenizer)


    def prepare_prompt(self, input_text: list):
        """
        Prepare the input text for the model by tokenizing it.

        :param input_text: The input code that needs to be repaired or completed.
        :return: Tokenized input ready for the model.
        """
        print(f"Input type: {type(input_text)}, value: {input_text}")

        if isinstance(input_text, list):
            # Extract the 'content' from each Message and join them into a single string
            input_text = " ".join([msg.content for msg in input_text if hasattr(msg, 'content')])

        # Tokenizing input text for the model
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        return inputs["input_ids"].to(self.model.device)

    def extract_output(self, output: torch.Tensor) -> str:
        """
        Extract the decoded output from the model.

        :param output: The raw output tensor from the model.
        :return: The decoded text as a string.
        """
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def generate_repair(self, input_text: str, max_tokens: int = 128, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        """
        Generate code repairs or completions for the input code.

        :param input_text: The broken code (e.g., with missing parts) that needs to be repaired.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: Sampling temperature for randomness in the output.
        :param num_comps: Number of code completions to generate.
        :return: The repaired code.
        """
        # Prepare the input prompt
        input_ids = self.prepare_prompt(input_text)

        # Generate the output using the model
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic outputs
            top_k=50,
            top_p=0.95,
            num_return_sequences=num_comps,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode and extract the outputs
        decoded_outputs = [self.extract_output(output) for output in outputs]
        
        # If only one completion, return the result as a string
        if num_comps == 1:
            return decoded_outputs[0]
        return decoded_outputs

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
