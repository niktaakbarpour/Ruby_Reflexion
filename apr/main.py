import os
import argparse
from immediate_refinement import run_immediate_refinement
from immediate_reflexion import run_immediate_reflexion
import time
import psutil
import pynvml
from simple import run_simple
from reflexion import run_reflexion
#from reflexion_ucs import run_reflexion_ucs
from test_acc import run_test_acc
from utils import read_jsonl, read_jsonl_gz
from run_reflexion_multi_agent import run_reflexion_multi_agent
from first_refl_omission import run_first_refl_omission
from self_refl_omission import run_self_refl_omission
from refl_omission import run_refl_omission
from test_gen_omission import run_test_gen_omission
from infer_spec import run_infer_spec

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`, `first_refl_omission`, `self_refl_omission`, `refl_omission`, `test_gen_omission`, `infer_spec`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs` or `rb`")
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)

    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")  # Temporary

    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    parser.add_argument("--model_path", type=str, help="Downloaded models path.")
    # parser.add_argument("--infer_spec", type=bool, help="Do infer specifications or not.")
    # TODO: implement this
    # parser.add_argument("--is_resume", action='store_true', help="To resume run")
    # parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "first_refl_omission":
        return kwargs_wrapper_gen(run_first_refl_omission, delete_keys=["expansion_factor"])
    elif strategy == "self_refl_omission":
        return kwargs_wrapper_gen(run_self_refl_omission, delete_keys=["expansion_factor"])
    elif strategy == "refl_omission":
        return kwargs_wrapper_gen(run_refl_omission, delete_keys=["expansion_factor"])
    elif strategy == "test_gen_omission":
        return kwargs_wrapper_gen(run_test_gen_omission, delete_keys=["expansion_factor"])
    elif strategy == "infer_spec":
        return kwargs_wrapper_gen(run_infer_spec, delete_keys=["expansion_factor"])
    elif strategy == "reflexion-multi":
        return kwargs_wrapper_gen(run_reflexion_multi_agent, delete_keys=["expansion_factor"])
    elif strategy == "immediate-reflexion":
        return kwargs_wrapper_gen(run_immediate_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "immediate-refinement":
        return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor"])
    elif strategy == "reflexion-ucs":
        return kwargs_wrapper_gen(run_reflexion_ucs)
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def log_resources(prefix=""):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f"{prefix} GPU memory: {gpu_mem:.2f} MB | CPU usage: {cpu}% | RAM usage: {ram:.2f} MB")

def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    # check if log path already exists
    # log_dir = os.path.join(args.root_dir, args.run_name)
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(log_dir, f"{args.language}.jsonl")
    # log_path = os.path.join(
    #     log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")
    # start the run
    # evaluate with pass@k
    start_time = time.time()
    log_resources("[START]")
    run_strategy(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        expansion_factor=args.expansion_factor,
        is_leetcode=args.is_leetcode,
        model_path=args.model_path,
        # infer_spec=args.infer_spec
    )

    end_time = time.time()
    log_resources("[END]")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)

# 'root/reflexion_deepseek_/rbugr-tiny._reflexion_3_deepseek-ai/deepseek-coder-6.7b-instruct_pass_at_k_3_rb.jsonl'