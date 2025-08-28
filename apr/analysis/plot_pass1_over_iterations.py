import json
import matplotlib.pyplot as plt

# List of input files and corresponding labels
file_info = [
    # ("../results/ds.jsonl", "RAMP"),
    ("../results/ds-self-omit.jsonl", "RAMP-self_refl"),
    # ("../results/ds-first-omit.jsonl", "RAMP-first_refl"),
    # ("../results/ds-refl-omit.jsonl", "RAMP-refl_refl"),
    # ("../results/ds-test-omit.jsonl", "RAMP-test_refl"),
]

num_iters = 11
plt.figure(figsize=(8, 5))

for file_path, label in file_info:
    # Tracks the number of samples solved cumulatively by each iteration
    success_cumulative = [0] * num_iters
    total_samples = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total_samples += 1
            first_success_iter = None

            # Find first iteration where pass@1_unit_iter{i} == 1.0
            for i in range(num_iters):
                if obj.get(f"pass@1_unit_iter{i}") == 1.0:
                    first_success_iter = i
                    break

            # Mark all iterations >= first_success_iter as successful
            if first_success_iter is not None:
                for j in range(first_success_iter, num_iters):
                    success_cumulative[j] += 1

    # Normalize by total number of samples
    pass_rates = [count / total_samples for count in success_cumulative]
    pass_rates = [p * 100 for p in pass_rates]
    plt.plot(range(num_iters), pass_rates, marker='o', linewidth=2, label=label)

# Customize the plot
plt.xticks(range(num_iters))
plt.ylim(40, 50)
plt.grid(True, linestyle=':', alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Pass@1")
plt.legend()
plt.tight_layout()
# plt.savefig("Cumulative_Pass@1_Unit.jpeg", format='jpeg')
plt.show()
