import json
import matplotlib.pyplot as plt

# Computes and plots the cumulative Pass@1 rate per iteration by finding the first successful repair in each JSONL record and marking all later iterations as solved, producing a curve from iteration 0 to 11.

# List of input files and corresponding labels
file_info = [
    ("", "RAMP"),
]

num_iters = 11   # actual repair iterations
plt.figure(figsize=(8, 5))

for file_path, label in file_info:
    total_samples = 0
    success_cumulative = [0] * (num_iters + 1)  # index 0 … 11

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total_samples += 1
            first_success_iter = None

            # Find first iteration where pass@1_unit_iter{i} == 1.0
            for i in range(num_iters):
                if obj.get(f"pass@1_unit_iter{i}") == 1.0:
                    first_success_iter = i + 1   # success reflected from iter 1
                    break

            # Mark all iterations >= first_success_iter as successful
            if first_success_iter is not None:
                for j in range(first_success_iter, num_iters + 1):
                    success_cumulative[j] += 1

    # Normalize by total number of samples
    pass_rates = [count / total_samples * 100 for count in success_cumulative]

    # Plot 0 … 11 (inclusive)
    plt.plot(range(num_iters + 1), pass_rates,
             marker='o', linewidth=3, color="#c97718", label=label)
print(pass_rates)

# Customize the plot
plt.xticks(range(num_iters + 1))
plt.xlim(left=0)
plt.ylim(bottom=0)   # force bottom-left origin

plt.grid(True, linestyle=':', alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Pass@1 (%)")
# plt.legend()

# Make axes cross at (0,0)
ax = plt.gca()
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# plt.savefig("pass_at_1_cumulative", dpi=200, bbox_inches="tight")

plt.show()
