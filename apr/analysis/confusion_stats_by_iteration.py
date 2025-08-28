# import json

# # --- Configuration ---
# jsonl_path = "../results/ds.jsonl"  # Change this as needed
# num_iters = 11  # Or however many iterations you use

# # --- Counters ---
# FN = [0] * num_iters
# FP = [0] * num_iters
# TN = [0] * num_iters
# TP = [0] * num_iters

# total_samples = 0

# with open(jsonl_path, "r", encoding="utf-8") as f:
#     for line in f:
#         entry = json.loads(line)
#         total_samples += 1

#         for i in range(num_iters):
#             key_iter = f"pass@1_iter{i}"
#             key_unit = f"pass@1_unit_iter{i}"

#             iter_val = entry.get(key_iter)
#             unit_val = entry.get(key_unit)

#             if iter_val is None or unit_val is None:
#                 continue  # skip if either is missing

#             iter_val = round(iter_val)  # convert 0.0/1.0 to int
#             unit_val = round(unit_val)

#             if iter_val == 0 and unit_val == 1:
#                 FN[i] += 1
#             elif iter_val == 1 and unit_val == 0:
#                 FP[i] += 1
#             elif iter_val == 0 and unit_val == 0:
#                 TN[i] += 1
#             elif iter_val == 1 and unit_val == 1:
#                 TP[i] += 1

# # --- Output ---
# print(f"Total samples: {total_samples}\n")
# for i in range(num_iters):
#     print(f"Iteration {i}:")
#     print(f"  FN = {FN[i]} ({FN[i]/total_samples:.3f})")
#     print(f"  FP = {FP[i]} ({FP[i]/total_samples:.3f})")
#     print(f"  TN = {TN[i]} ({TN[i]/total_samples:.3f})")
#     print(f"  TP = {TP[i]} ({TP[i]/total_samples:.3f})")
#     print()

import json
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
jsonl_path = "../results/ds.jsonl"  # Change as needed
num_iters = 11

# --- Initialize counts ---
TP = [0] * num_iters
FP = [0] * num_iters
FN = [0] * num_iters
TN = [0] * num_iters
total_samples = 0

# --- Read and classify ---
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        total_samples += 1

        for i in range(num_iters):
            iter_key = f"pass@1_iter{i}"
            unit_key = f"pass@1_unit_iter{i}"

            iter_val = entry.get(iter_key)
            unit_val = entry.get(unit_key)

            if iter_val is None or unit_val is None:
                continue

            iter_val = round(iter_val)
            unit_val = round(unit_val)

            if iter_val == 0 and unit_val == 1:
                FN[i] += 1
            elif iter_val == 1 and unit_val == 0:
                FP[i] += 1
            elif iter_val == 0 and unit_val == 0:
                TN[i] += 1
            elif iter_val == 1 and unit_val == 1:
                TP[i] += 1

# --- Normalize to proportions ---
TP_rate = [v / total_samples for v in TP]
FP_rate = [v / total_samples for v in FP]
FN_rate = [v / total_samples for v in FN]
TN_rate = [v / total_samples for v in TN]

# --- Plot ---
x = np.arange(num_iters)
bar_width = 0.6

plt.figure(figsize=(10, 6))
plt.bar(x, TN_rate, label="TN", color="lightgray")
plt.bar(x, FN_rate, bottom=TN_rate, label="FN", color="orange")
plt.bar(x, FP_rate, bottom=np.array(TN_rate) + np.array(FN_rate), label="FP", color="salmon")
plt.bar(x, TP_rate, bottom=np.array(TN_rate) + np.array(FN_rate) + np.array(FP_rate), label="TP", color="seagreen")

plt.xticks(x, [f"Iter {i}" for i in x])
plt.xlabel("Iteration")
plt.ylabel("Proportion of Samples")
plt.title("Confusion Categories (TP / FP / FN / TN) per Iteration")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
# plt.savefig("confusion_analysis_per_iteration.png", dpi=300)
plt.show()
