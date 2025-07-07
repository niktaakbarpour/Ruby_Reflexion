import json
import matplotlib.pyplot as plt

file_path = "../results/reflexion_edge_pass_at1_iter11.jsonl"  # Make sure this matches your actual file name

success_cumulative = [0] * 11  # Iter 0 to 10
total_objects = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        total_objects += 1

        # Track if any earlier iteration passed
        solved = False
        for i in range(11):
            key = f"pass@1_unit_iter{i}"
            if not solved and obj.get(key) == 1.0:
                # Mark this iteration and all future ones as successful
                for j in range(i, 11):
                    success_cumulative[j] += 1
                solved = True  # Only count first successful iter
                break

print("Total objects:", total_objects)
print("Cumulative success counts per iteration:", success_cumulative)

# Compute cumulative pass@1 rates
pass_rates = [count / total_objects for count in success_cumulative]
print("Cumulative pass@1 rates:", pass_rates)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(range(11), pass_rates, marker='o', linewidth=2, label='Ruby')
plt.xticks(range(11))
plt.ylim(min(pass_rates) - 0.01, 0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Pass@1")
plt.title("Cumulative Pass@1 over Iterations")
plt.legend()
plt.tight_layout()
plt.show()


