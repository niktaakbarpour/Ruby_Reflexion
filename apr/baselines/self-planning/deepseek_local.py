from typing import List, Union, Optional, Literal, Dict
import dataclasses
import re
from transformers import StoppingCriteria, StoppingCriteriaList

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])



class StreamTokens(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the last token
        token_id = input_ids[0, -1]
        print(self.tokenizer.decode([token_id], skip_special_tokens=True), end='', flush=True)
        return False  # Return False to continue generating



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
        tokens = self.prepare_prompt(messages).to(self.model.device)
        
        stream_callback = StreamTokens(self.tokenizer)

        # Generate model outputs
        with torch.inference_mode():
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                eos_token_id=self.eos_token_id,
                num_return_sequences=num_comps,
                # use_cache=False,
                stopping_criteria=StoppingCriteriaList([stream_callback])
            )
        print(f"output_tokens: {output_tokens}")
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
        print(f"decoded_outputs: {decoded_outputs}")
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

        # Initialize token usage counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0


        """
        Initialize the DeepSeekCoder model with 4-bit NF4 quantization.
        """

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )


        model = AutoModelForCausalLM.from_pretrained(
            model_path if model_path is not None else "deepseek-ai/deepseek-coder-6.7b-instruct",
            quantization_config=nf4_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path is not None else "deepseek-ai/deepseek-coder-6.7b-instruct",
            trust_remote_code=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        print(f"Model loaded on device: {device}")
        print(f"Actual model device: {next(model.parameters()).device}")
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
        )


    def generate_chat(self, messages, temperature=0.0, num_comps=1):
        """
        Override to add token usage tracking.
        """
        # Count prompt tokens
        # prompt_text = self.tokenizer.apply_chat_template(
        #     [{"role": m.role, "content": m.content} for m in messages],
        #     add_generation_prompt=True
        # )
        # prompt_tokens = len(self.tokenizer.encode(prompt_text))
        # prompt_tokens = len(self.tokenizer.encode(str(prompt_text)))

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            add_generation_prompt=True,
            tokenize=False  # Ensure output is string
        )
        prompt_tokens = len(self.tokenizer.encode(prompt_text))


        self.total_prompt_tokens += prompt_tokens

        # Generate using parent method
        result = super().generate_chat(messages, temperature=temperature, num_comps=num_comps)

        # Count completion tokens
        if "choices" in result and len(result["choices"]) > 0:
            output_text = result["choices"][0]["message"]["content"]
            completion_tokens = len(self.tokenizer.encode(output_text))
        else:
            completion_tokens = 0
        self.total_completion_tokens += completion_tokens

        # Add usage field like OpenAI API
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return result


    def extract_output(self, output: str) -> str:
        print("text")
        text = output.split("### Response:", 1)[-1] if "### Response:" in output else output
        text = text.replace("<|EOT|>", "").replace("<｜begin▁of▁sentence｜>", "").strip()

        # Keep only code block if present
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                # Return the content of the first code block
                return f"```{parts[1]}```".strip()
        print(f"text final: {text}")
        return text
    

    def get_total_tokens(self):
        """Return total tokens used so far."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }