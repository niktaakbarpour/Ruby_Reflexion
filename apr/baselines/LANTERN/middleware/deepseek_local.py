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
    
# class HFModelBase(ModelBase):
#     """
#     Base for huggingface chat models
#     """

#     def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
#         self.name = model_name
#         self.model = model
#         self.tokenizer = tokenizer
#         self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
#         self.is_chat = True

#     def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
#         import torch
#         # NOTE: HF does not like temp of 0.0.
#         if temperature < 0.0001:
#             temperature = 0.0001

#         prompt = self.prepare_prompt(messages)

#         outputs = self.model.generate(
#             prompt,
#             max_new_tokens=min(
#             max_tokens, self.model.config.max_position_embeddings),
#             use_cache=True,
#             # do_sample=True,
#             temperature=temperature,
#             top_p=1.0,
#             eos_token_id=self.eos_token_id,
#             num_return_sequences=num_comps,
#         )

#         if isinstance(outputs, torch.Tensor):
#             outputs = outputs.tolist()  # Convert tensor to list of token IDs

#         # Decode outputs
#         outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
#         # Ensure `outs` is a list of strings
#         assert isinstance(outs, list), f"Expected outs to be a list, got {type(outs)}"
#         for i, out in enumerate(outs):
#             assert isinstance(out, str), f"Expected out to be a string, got {type(out)}"

#         # If you need additional processing, call extract_output only if necessary
#         # Assuming extract_output applies further processing on decoded strings:
#         outs = [self.extract_output(out) for out in outs]
#         # Return the decoded output(s)
#         if len(outs) == 1:
#             return outs[0]  # Return a single string
#         else:
#             return outs  # Return a list of strings

#     def prepare_prompt(self, messages: List[Message]):
#         raise NotImplementedError

#     def extract_output(self, output: str) -> str:
#         raise NotImplementedError

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
        decoded_outputs = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
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



    # def extract_output(self, output: str) -> str:
    #     text = output.split("### Response:", 1)[-1] if "### Response:" in output else output
    #     text = text.replace("<|EOT|>", "").replace("<｜begin▁of▁sentence｜>", "").strip()

    #     # Keep only code block if present
    #     if "```" in text:
    #         parts = text.split("```")
    #         if len(parts) >= 2:
    #             # Return the content of the first code block
    #             return f"```{parts[1]}```".strip()
    #     return text
    


    @staticmethod
    def _strip_special_tokens(text: str) -> str:
        SPECIAL_TOKEN_PATTERNS = [
            r"<\|EOT\|>",
            r"<\|endoftext\|>",
            r"<｜begin▁of▁sentence｜>",
            r"<｜end▁of▁sentence｜>",
        ]
        for pat in SPECIAL_TOKEN_PATTERNS:
            text = re.sub(pat, "", text)
        text = re.sub(r"<\|[^>]*\|>", "", text)    # remove <| ... |>
        text = re.sub(r"<｜[^>]*｜>", "", text)    # remove <｜ ... ｜>
        return text.strip()

    def extract_output(self, output: str) -> str:
        text = output.split("### Response:", 1)[-1] if "### Response:" in output else output
        text = self._strip_special_tokens(text)

        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return f"```{parts[1]}```".strip()
        return text.strip()




# class DeepSeekCoder(HFModelBase):
#     import torch
#     def __init__(self, model_path=None):
#         from typing import List, Union
#         from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#         import torch
        
#         """
#         Initialize the DeepseekCoder model.

#         :param model_path: local path to the model if you have downloaded it locally.
#         """

#         nf4_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 to save memory
#             bnb_4bit_use_double_quant=True    # Double quantization for memory efficiency
#         )

#         print(torch.cuda.memory_summary(device=None, abbreviated=True))

#         model = AutoModelForCausalLM.from_pretrained(
#             model_path if model_path is not None else "deepseek-ai/deepseek-coder-6.7b-instruct",
#             quantization_config=nf4_config,
#             trust_remote_code=True,
#             torch_dtype=torch.float16,   # Ensures efficient memory usage
#             device_map="auto"            # Automatically assigns layers to GPU
#         )


#         # Load the tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
#             trust_remote_code=True
#         )
#         model.config.use_cache = False
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Model loaded on device: {device}")
#         print(torch.cuda.memory_summary(device=None, abbreviated=True))

#         # model.to(device)  # not strictly needed with `load_in_4bit=True`, but safe
#         super().__init__("deepseek-ai/deepseek-coder-6.7b-instruct", model, tokenizer)

#     def prepare_prompt(self, messages: List[Message]):
#         bos_token = "<|begin▁of▁sentence|>"
#         eos_token = "<|end▁of▁sentence|>"
#         prompt = bos_token

#         for message in messages:
#             if message.role == "system":
#                 prompt += f"{message.content}\n\n"
#             elif message.role == "user":
#                 prompt += f"User: {message.content}\n\n"
#             elif message.role == "assistant":
#                 prompt += f"Assistant: {message.content}{eos_token}\n\n"

#         return prompt



    # def extract_output(self, output: str) -> str:
    #     """
    #     Clean and extract the main code content from the model output.
    #     Works even without fenced code blocks.
    #     """

    #     SPECIAL_TOKEN_ANY_RE = re.compile(r"(?:<\|[^|>]*\|>|\|[^|>]*\|>)")
    #     DANGLING_FRAGMENT_RE = re.compile(r"(?:sentence\|>|^\|>| \|>)")

    #     # Step 1: Normalize DeepSeek characters
    #     output = output.replace("▁", " ")
    #     # Step 2: Remove special tokens
    #     output = SPECIAL_TOKEN_ANY_RE.sub("", output)
    #     output = DANGLING_FRAGMENT_RE.sub("", output)
    #     output = output.replace("<|begin of sentence|>", "")
    #     output = output.replace("<|end of sentence|>", "")
    #     # Step 3: If fenced code exists, extract it
    #     if "```" in output:
    #         parts = output.split("```")
    #         for i in range(1, len(parts), 2):
    #             code_block = parts[i].strip()
    #             if "\n" in code_block:
    #                 first_line = code_block.splitlines()[0].strip()
    #                 if re.match(r"^[a-zA-Z]+$", first_line):  # language line like "ruby"
    #                     code_block = "\n".join(code_block.splitlines()[1:])
    #             return code_block.strip()

    #     # Step 4: Fallback — detect code-like lines
    #     lines = output.splitlines()
    #     code_lines = []
    #     code_started = False

    #     for line in lines:
    #         stripped = line.strip()
    #         # A line is "code-like" if it matches Ruby syntax patterns
    #         if re.search(r"[=+\-*/%<>]|def |class |puts|end\b|\{|}", stripped):
    #             code_started = True
    #             code_lines.append(stripped)
    #         elif code_started:
    #             # Stop when NL text appears after code
    #             if stripped == "" or not re.search(r"[=+\-*/%<>]|def |class |puts|end\b|\{|}", stripped):
    #                 break

    #     code = "\n".join(code_lines).strip()
    #     # Step 5: Remove a language line if present (e.g., "ruby")
    #     code_lines = code.splitlines()
    #     if len(code_lines) > 1 and re.match(r"^[a-zA-Z]+$", code_lines[0].strip()):
    #         code = "\n".join(code_lines[1:])
    #     # Step 6: Normalize escaped newlines (only if '\\n' exists)
    #     if "\\n" in code:
    #         code = code.replace("\\n", "\n")

    #     return code.strip()

