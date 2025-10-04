from middleware.coordinator import Coordinator
import argparse
from middleware.deepseek_local import DeepSeekCoder
import os
import torch
import time
import psutil
import pynvml
import torch.multiprocessing as mp

if __name__ == "__main__":
    # torch.cuda.empty_cache()
    start_time = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    mp.set_start_method("spawn", force=True)
    def log_resources(prefix=""):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.Process().memory_info().rss / 1024**2
        print(f"{prefix} GPU memory: {gpu_mem:.2f} MB | CPU usage: {cpu}% | RAM usage: {ram:.2f} MB")
    log_resources("[START]")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    llm = DeepSeekCoder("")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/tr.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        '--restart',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="True: restart the process from the beginning. False: resume from the latest state."
    )
    args = parser.parse_args()

    cd = Coordinator(args.config, args.restart, llm=llm)
    cd.run()
    end_time = time.time()
    log_resources("[END]")
    print(f"Total time: {end_time - start_time:.2f} seconds")

