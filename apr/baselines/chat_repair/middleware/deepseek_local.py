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


# --- context-preprocessing helpers ------------------------------------------

# Matches ```lang\n ... \n``` or ```\n ... \n```
FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)

def _to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(map(str, x))
    return str(x)


def clamp_blocks(text: str, max_chars: int = 4000) -> str:
    def _clamp_block(block: str) -> str:
        if len(block) <= max_chars:
            return block
        head = block[: max_chars // 2]
        tail = block[- max_chars // 2 :]
        return head + f"\n/* ... {len(block)-max_chars} chars omitted ... */\n" + tail

    def _repl_fence(m: re.Match) -> str:
        lang = m.group(1) or ""
        inner = m.group(2)
        return f"```{lang}\n{_clamp_block(inner)}```"
    text = _to_text(text)
    text = FENCE_RE.sub(_repl_fence, text)
    text = re.sub(r"[^\n]{8000,}", lambda m: _clamp_block(m.group(0)), text)
    return text


def minify_old_turn(m: Message) -> Message:
    c = _to_text(m.content)  # <— normalize first
    if m.role in ("user", "assistant"):
        c = re.sub(r"```.*?```", "```// code omitted (history) ```", c, flags=re.S)
        return Message(role=m.role, content=c)
    return Message(role=m.role, content=c)



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


    def _fit_messages_to_ctx(self, messages, reserve):
        real_limit = getattr(self.model.config, "max_position_embeddings", None)
        if real_limit is None or real_limit > 10**9:
            real_limit = getattr(self.tokenizer, "model_max_length", 16384)
        self.tokenizer.model_max_length = real_limit
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"

        def toklen(msgs):
            ids = self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in msgs],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            return len(ids)

        # If there is no user turn, just trim from the left as needed.
        user_idxs = [i for i, m in enumerate(messages) if m.role == "user"]
        if not user_idxs:
            trimmed = list(messages)
            ctx_budget = max(0, real_limit - max(0, reserve))
            # drop oldest until it fits
            while toklen(trimmed) > ctx_budget and len(trimmed) > 1:
                trimmed.pop(0)
            return trimmed, real_limit

        # Anchor set: first system (optional), last assistant before last user (optional), last user (required)
        system = next((m for m in messages if m.role == "system"), None)
        last_user_idx = max(user_idxs)
        last_user = messages[last_user_idx]

        last_assistant = None
        for i in range(last_user_idx - 1, -1, -1):
            if messages[i].role == "assistant":
                last_assistant = messages[i]
                break

        trimmed = []
        if system: trimmed.append(system)
        if last_assistant: trimmed.append(last_assistant)
        trimmed.append(last_user)

        seen = set(id(m) for m in trimmed)
        ctx_budget = max(0, real_limit - max(0, reserve))

        # Add earlier messages from newest→oldest excluding already included
        for m in reversed(messages):
            if id(m) in seen:
                continue
            candidate = ([system] if system else []) + ([last_assistant] if last_assistant else []) + [m, last_user]
            if toklen(candidate) <= ctx_budget:
                trimmed = candidate
                seen.add(id(m))
            else:
                break

        return trimmed, real_limit



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





        # --- PREPROCESS CONTEXT ---
        if messages:
            # Ensure every message.content is str for preprocess
            messages = [Message(m.role, _to_text(m.content)) for m in messages]
            user_idxs = [i for i, m in enumerate(messages) if m.role == "user"]
            if user_idxs:
                last_user_idx = user_idxs[-1]

                # find last assistant before last user (optional)
                last_asst_idx = None
                for i in range(last_user_idx - 1, -1, -1):
                    if messages[i].role == "assistant":
                        last_asst_idx = i
                        break

                pre = []
                for i, m in enumerate(messages):
                    if i == last_user_idx:
                        pre.append(Message(role=m.role, content=clamp_blocks(m.content)))
                    elif last_asst_idx is not None and i == last_asst_idx:
                        pre.append(m)
                    else:
                        pre.append(minify_old_turn(m))
                messages = pre
            else:
                # no user turn → just minify old turns to be safe
                messages = [minify_old_turn(m) for m in messages]
# ---------------------------

        reserve = max_tokens + 128
        messages, real_limit = self._fit_messages_to_ctx(messages, reserve)
        tokens = self.prepare_prompt(messages)
        if hasattr(tokens, "dim") and tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        if tokens.shape[1] > real_limit - max(0, reserve):
            tokens = tokens[:, -(real_limit - reserve):]

        # Generate model outputs
        output_tokens = self.model.generate(
            tokens,
            max_new_tokens=min(max_tokens, getattr(self.model.config, "max_position_embeddings", real_limit)),
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
        # choices = [{"message": {"role": "assistant", "content": text}} for text in processed_outputs]
        # return {"choices": choices, "prompt": messages[-1].content}
        choices = [{"message": {"role": "assistant", "content": text}} for text in processed_outputs]
        last_content = ""
        if messages:
            # find last user content if any, else last message content
            user_idxs = [i for i, m in enumerate(messages) if m.role == "user"]
            if user_idxs:
                last_content = messages[user_idxs[-1]].content
            else:
                last_content = messages[-1].content
        return {"choices": choices, "prompt": last_content}


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

        # Set pad token if missing
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Optional: also set eos for completeness
        if getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id


        real_limit = getattr(model.config, "max_position_embeddings", None)
        if real_limit and real_limit < 10**9:
            tokenizer.model_max_length = real_limit
        if hasattr(tokenizer, "truncation_side"):
            tokenizer.truncation_side = "left"

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

