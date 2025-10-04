import yaml
from repairer import gen_apr, re_gen
from translator import initilize, translate, back_translate
from analyzer import decide
from evaluator import eval_apr, get_result
from . import history
import logging
import os
import time
import psutil
import pynvml

class Coordinator:
    def __init__(self, config_path, restart, llm):
        self.llm = llm
        self.config_path = config_path
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S", 
        handlers=[
            logging.FileHandler(os.path.join(self.config["log_dir"], f"{self.config['name']}_logs.txt")), 
            logging.StreamHandler() 
            ]
        )
    
    def __get_args(self, *args):
        res = {}
        for arg in args:
            if ":" in arg:
                k, v = arg.split(":")
                res[k] = int(v) if v.isnumeric() else v
            else:
                v = None
                for k in arg.split("."):
                    if v is None:
                        v = self.config[k]
                    else:
                        v = v[k]
                res[arg.split(".")[-1]] = v
        return res
    
    def __get_repair_mode(self):
        return self.config["repair"]["mode"]
    
    def __check_termination(self):
        '''
        Check termination conditions.
        '''
        if self.__get_state("it") >= self.__get_args("termination.max_it")["max_it"]:
            return True
        return False
    
    def __check_run(self, action, condition, kwarg):
        '''
        Execute the action if the condition is satisfied.
        '''
        if condition:
            action(**kwarg)
            return True
        return False
    
    def __perform_action(self, action, condition, it, action_name, desc, kwarg):
        self.__log_record(it, action_name, desc)
        return self.__check_run(action, condition, kwarg)
    
    def __get_state(self, k):
        return self.config["state"][k]
    
    def __update_state(self, it=-1, action=None):
        if it != -1:
            self.config["state"]["it"] = it
        if action is not None:
            self.config["state"]["action"] = action
        self.__update_config()
    
    def __update_config(self):
        with open(self.config_path, "w") as file:
            yaml.dump(self.config, file)
    
    def __check_state(self, k, v, op="eq"):
        if op == "eq":
            return self.config["state"][k] == v
        elif op == "ge":
            return self.config["state"][k] >= v
        elif op == "le":
            return self.config["state"][k] <= v
        elif op == "g":
            return self.config["state"][k] > v
        elif op == "l":
            return self.config["state"][k] < v
        elif op == "in":
            return self.config["state"][k] in v
        else:
            return False
    
    def __check_mode(self, mode, op="eq"):
        if op == "eq":
            return self.config["translate"]["mode"] == mode
        elif op == "not":
            return self.config["translate"]["mode"] != mode
        elif op == "in":
            return self.config["translate"]["mode"] in mode
        else:
            return False
        

    def _base_run(self):
        # Snapshot to avoid cascading through multiple steps in one call
        start_it = self.__get_state("it")
        start_action = self.__get_state("action")

        # Step 1: gen_apr
        if start_it == 0 and start_action == "start":
            if self.__perform_action(
                gen_apr.run,
                True,  # we already checked the condition above
                0,
                "gen_apr",
                "generating patched code",
                {**self.__get_args("base_dir", "num_proc", "dry_run", "gen.nsample", "gen.nattempt", "gen.temperature", "dataset_path"), "llm": self.llm}
            ):
                self.__update_state(action="gen")
                return  # stop here; next step will run on the next invocation

        # Step 2: eval_apr
        if start_it == 0 and start_action == "gen":
            if self.__perform_action(
                eval_apr.run,
                True,
                0,
                "eval_apr",
                "evaluating patched code",
                {**self.__get_args("base_dir", "state.it", f"mode:{self.config['repair']['mode']}"), "llm": self.llm}
            ):
                self.__update_state(action="eval")
                return

        # Step 3: get_result
        if start_it == 0 and start_action == "eval":
            if self.__perform_action(
                get_result.run,
                True,
                0,
                "cal",
                "calculating results",
                {**self.__get_args("base_dir", "result.k", "state.it", "name", "note:base run"), "llm": self.llm}
            ):
                self.__update_state(action="cal")
                return

        # Step 4: history
        if start_it == 0 and start_action == "cal":
            if self.__perform_action(
                history.build_history,
                True,
                0,
                "save_history",
                "saving historical data",
                self.__get_args("base_dir", "it:0")
            ):
                self.__update_state(action="save_history")
                return

    def __log_record(self, it=0, action="start", desc=""):
        print(f"[DIAG] logging it={it}, persisted it={self.config['state']['it']}")
        if desc == "":
            content = f"Iteration {it}: Performing action <{action}>"
        else:
            content = f"Iteration {it}: Performing action <{action}> - {desc}"
        logging.info(content)
    
    
    def run(self):
        print("Coordinator.run() started")
        start_time = time.time()

        # ---- Safe resource logger (NVML optional) ----
        def log_resources(prefix=""):
            try:
                import pynvml as _nvml
                _nvml.nvmlInit()
                handle = _nvml.nvmlDeviceGetHandleByIndex(0)
                gpu_mem = _nvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                _nvml.nvmlShutdown()
                gpu_str = f"{gpu_mem:.2f} MB"
            except Exception:
                gpu_str = "N/A"
            try:
                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.Process().memory_info().rss / 1024**2
            except Exception:
                cpu, ram = "N/A", "N/A"
            print(f"{prefix} GPU: {gpu_str} | CPU: {cpu}% | RAM: {ram} MB")

        # One step at a time for the very first iteration (prelude)
        print("_base_run() called")
        log_resources("[START]")
        self._base_run()

        # ---- Main state machine loop ----
        while not self.__check_termination():

            it = self.__get_state("it")
            action = self.__get_state("action")
            this_it = it + 1  # <-- compute the next-iteration index up front

            print(f"[LOOP] it={it}, action={action}")
            print(f"[STATE] it={self.__get_state('it')} action={self.__get_state('action')}")

            # Optional fast-exit for diff mode
            if self.__check_mode("diff") and it >= 1:
                print('############################ tr diff finished ############################')
                break

            progressed = False  # track whether any step succeeded this pass

            print(f"[STATE] it={self.__get_state('it')} action={self.__get_state('action')}")

            # === initialize (start new iteration after base prelude) ===
            if self.__check_state("it", 0, "ge") and self.__check_state("action", ["save_history"], "in"):
                print(f"[DIAG] persisted state before initilize: it={self.__get_state('it')}, action={self.__get_state('action')}")

                if self.__perform_action(
                    initilize.run,
                    True,
                    this_it,
                    "initialize",
                    "initializing new iteration",
                    self.__get_args("base_dir", f"it:{this_it}", "unfixed_k")
                ):
                    print("initialize.run -> OK")
                    self.__update_state(action="init")
                    progressed = True

            # === decide ===
            elif self.__check_state("action", "init") and self.__check_mode(["reasoning", "nohist", "nocot"], "in"):
                print(f"[DIAG] persisted state before decide: it={self.__get_state('it')}, action={self.__get_state('action')}")

                if self.__perform_action(
                    decide.run,
                    True,
                    this_it,
                    "decide",
                    "determining target language",
                    {**self.__get_args("base_dir", "num_proc", "dry_run", f"it:{this_it}", "translate.mode", "hist_top_k", "dataset_path"), "llm": self.llm}
                ):
                    print("decide.run -> OK")
                    self.__update_state(action="decide")
                    progressed = True

            # === translate ===
            elif self.__check_state("action", ["init", "decide"], "in") and self.__check_mode("notrans", "not"):
                print(f"[DIAG] persisted state before translate: it={self.__get_state('it')}, action={self.__get_state('action')}")

                if self.__perform_action(
                    translate.run,
                    True,
                    this_it,
                    "translate",
                    "translating unfixed bugs",
                    {**self.__get_args("base_dir", "num_proc", "dry_run", f"it:{this_it}", "translate.mode", f"r_mode:{self.__get_repair_mode()}", "dataset_path", f"config_path:{self.config_path}"), "llm": self.llm}
                ):
                    print("translate.run -> OK")
                    self.__update_state(action="translate")
                    progressed = True

            # === re_gen ===
            elif self.__check_state("action", ["translate", "init"], "in"):
                print(f"[DIAG] persisted state before re_gen: it={self.__get_state('it')}, action={self.__get_state('action')}")

                if self.__perform_action(
                    re_gen.run,
                    True,
                    this_it,
                    "re_gen",
                    "generating patched code for unfixed bugs",
                    {**self.__get_args("base_dir", "num_proc", "dry_run", "gen.nsample", "gen.nattempt", f"it:{this_it}", "repair.mode", "gen.temperature", "dataset_path"), "llm": self.llm}
                ):
                    print("re_gen.run -> OK")
                    self.__update_state(action="re_gen")
                    progressed = True

            # === back_translate ===
            elif self.__check_state("action", "re_gen") and self.__check_mode("notrans", "not"):
                print(f"[DIAG] persisted state before back_translate: it={self.__get_state('it')}, action={self.__get_state('action')}")

                if self.__perform_action(
                    back_translate.run,
                    True,
                    this_it,
                    "back_translate",
                    "back-translating generated patched code",
                    {**self.__get_args("base_dir", "num_proc", "dry_run", f"it:{this_it}", "repair.mode"), "llm": self.llm}
                ):
                    print("back_translate.run -> OK")
                    self.__update_state(action="back_translate")
                    progressed = True

            # === re_eval ===
            elif self.__check_state("action", ["back_translate", "re_gen"], "in"):
                print(f"[DIAG] persisted state before eval_apr: it={self.__get_state('it')}, action={self.__get_state('action')}")
                break
                # if self.__perform_action(
                #     eval_apr.run,
                #     True,
                #     this_it,
                #     "re_eval",
                #     "evaluating back-translated patched code",
                #     {**self.__get_args("base_dir", f"it:{this_it}", "repair.mode"), "llm": self.llm}
                # ):
                #     print("re_eval.run -> OK")
                #     self.__update_state(action="re_eval")
                #     progressed = True

            # # === re_cal ===
            elif self.__check_state("action", "re_eval"):
                print(f"[DIAG] persisted state before get_result: it={self.__get_state('it')}, action={self.__get_state('action')}")
                break
                # if self.__perform_action(
                #     get_result.run,
                #     True,
                #     this_it,
                #     "re_cal",
                #     "calculating results",
                #     {**self.__get_args("base_dir", "result.k", f"it:{this_it}", "name", f"note:iter {this_it}"), "llm": self.llm}
                # ):
                #     print("re_cal.run -> OK")
                #     self.__update_state(action="re_cal")
                #     if self.__check_mode("notrans", "eq"):
                #         self.__update_state(it=this_it)
                #     progressed = True

            # === save_history -> only after this, increment it ===
            elif self.__check_state("action", "re_cal") and self.__check_mode("notrans", "not"):
                print(f"[STATE 2] it={self.__get_state('it')} action={self.__get_state('action')}")
                print(f"[DIAG] persisted state before build_history: it={self.__get_state('it')}, action={self.__get_state('action')}")
                if self.__perform_action(
                    history.build_history,
                    True,
                    this_it,
                    "save_history",
                    "saving historical data",
                    self.__get_args("base_dir", f"it:{this_it}")
                ):
                    print("save_history.run -> OK")
                    # Advance to next iteration and set action so the next pass starts at initialize
                    self.__update_state(it=this_it, action="save_history")
                    progressed = True
                    
            log_resources("[LOOP-END]")
            print(f"Elapsed: {time.time() - start_time:.2f}s")
            # === Unconditional iteration bump (like your example) ===
            # self.__update_state(it=this_it)

            # If nothing progressed this pass, we’re either waiting for external state or done.
            if not progressed:
                if self.__check_termination():
                    break
                # Avoid a hot loop if guards are temporarily false
                time.sleep(0.5)



        print("Coordinator.run() finished")


                
