from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        print('python here')
        return PyGenerator()
    elif lang == "rb" or lang == "ruby":
        print('ruby here')
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        print('rust here')
        return RsGenerator()
    else:
        print('nowhere here')
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str, model_path:str = None) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        if model_path is not None:
            kwargs["model_path"] = model_path
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
