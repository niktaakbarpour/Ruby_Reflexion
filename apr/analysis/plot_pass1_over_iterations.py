import json
import matplotlib.pyplot as plt

# List of input files and corresponding labels
file_info = [
    ("../results/reflexion_edge_pass_at1_iter11.jsonl", "RAMP"),
    ("../results/self_refl_omit_edge_pass_at1_iter11.jsonl", "RAMP-self_refl"),
    ("../results/first_refl_omit_edge_pass_at1_iter11.jsonl", "RAMP-self_first")
]

num_iters = 11
plt.figure(figsize=(8, 5))

for file_path, label in file_info:
    success_cumulative = [0] * num_iters
    total_objects = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total_objects += 1

            solved = False
            for i in range(num_iters):
                key = f"pass@1_unit_iter{i}"
                if not solved and obj.get(key) == 1.0:
                    for j in range(i, num_iters):
                        success_cumulative[j] += 1
                    solved = True
                    break

    pass_rates = [count / total_objects for count in success_cumulative]
    plt.plot(range(num_iters), pass_rates, marker='o', linewidth=2, label=label)

# Customize plot
plt.xticks(range(num_iters))
plt.ylim(0.15, 0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Pass@1")
plt.legend()
plt.tight_layout()
# plt.savefig("Cumulative_Comparison.png", format='png')
plt.show()
