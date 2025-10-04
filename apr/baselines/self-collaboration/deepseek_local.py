from typing import List, Union, Optional, Literal, Dict
import dataclasses
import re


MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

model_path = ""

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

class HFModelBase(ModelBase):
    """
    Base for Hugging Face chat models.
    This version mimics OpenAI's ChatCompletion response format.
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0,
        num_comps: int = 1
    ) -> Dict:
        """
        Generates responses in a format similar to OpenAI ChatCompletion:
        {
          "choices": [{"message": {"role": "assistant", "content": ...}}],
          "prompt": ...
        }
        """
        import torch

        # Ensure temperature is not zero
        if temperature < 0.0001:
            temperature = 0.0001

        # Prepare input tokens from chat messages
        tokens = self.prepare_prompt(messages)

        # Generate model outputs
        output_tokens = self.model.generate(
            tokens,
            max_new_tokens=min(max_tokens, self.model.config.max_position_embeddings),
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
            use_cache=True,
        )

        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
        processed_outputs = [self.extract_output(out) for out in decoded_outputs]

        # Construct OpenAI-style response
        choices = [{"message": {"role": "assistant", "content": text}} for text in processed_outputs]
        return {"choices": choices, "prompt": messages[-1].content}

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class DeepSeekCoder(HFModelBase):
    def __init__(self, model_path=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch


        import os
        from pathlib import Path

        REQUIRED_FILES = {"config.json", "tokenizer.json"}  # add more if your folder has them

        def _resolve_local_model_dir(p: str | os.PathLike) -> Path:
            """
            Return a Path to the directory that actually contains config.json.
            Raise a helpful error if it doesn't look like a HF model folder.
            """
            p = Path(p).expanduser().resolve()
            if p.is_file():
                # e.g., user passed a file inside the dir; go up to its parent
                p = p.parent
            # Some zips or extra nesting like .../deepseek/XXX/<actual-model-dir>
            # If config.json isn't here, search one level down for a dir that has it.
            if not (p / "config.json").exists():
                for sub in p.iterdir():
                    if sub.is_dir() and (sub / "config.json").exists():
                        return sub
                missing = ", ".join(sorted(REQUIRED_FILES))
                raise FileNotFoundError(
                    f"Could not find a model directory with {missing} under: {p}"
                )
            return p



        """
        Initialize the DeepSeekCoder model with 4-bit NF4 quantization.
        """
        local_dir = _resolve_local_model_dir(
            model_path if model_path is not None else ""
        )

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 to save memory
            bnb_4bit_use_double_quant=True    # Double quantization for memory efficiency
        )


        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            local_files_only=True, 
            # model_path if model_path is not None else "deepseek-ai/deepseek-coder-6.7b-instruct",
            quantization_config=nf4_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,   # Ensures efficient memory usage
            # device_map="auto"            # Automatically assigns layers to GPU
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,  
            # model_path if model_path is not None else "deepseek-ai/deepseek-coder-6.7b-instruct",
            trust_remote_code=True
        )
        if torch.cuda.is_available():
            model.to("cuda")

        model.config.use_cache = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded on device: {device}")
        print(torch.cuda.memory_summary(device=None, abbreviated=True))

        super().__init__("deepseek-ai/deepseek-coder-6.7b-instruct", model, tokenizer)

    def prepare_prompt(self, messages: List['Message']):
        formatted_messages = [
            {"role": m.role, "content": m.content if isinstance(m.content, str) else "\n".join(m.content)}
            for m in messages
        ]
        return self.tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)



    def extract_output(self, output: str) -> str:
        text = output.split("### Response:", 1)[-1] if "### Response:" in output else output
        text = text.replace("<|EOT|>", "").replace("<｜begin▁of▁sentence｜>", "").strip()

        # Keep only code block if present
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                # Return the content of the first code block
                return f"```{parts[1]}```".strip()
        return text

