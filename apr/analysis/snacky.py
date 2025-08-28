# import json
# import pandas as pd
# import plotly.graph_objects as go

# # Load JSONL
# jsonl_path = "../results/ds.jsonl"
# data = []
# with open(jsonl_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         data.append(json.loads(line))

# # Build transition counts between each consecutive iteration
# label_set = ["Incorrect", "Correct"]
# label_index = {"Incorrect": 0, "Correct": 1}
# sources, targets, values, labels = [], [], [], []
# counts = {}

# for i in range(10):  # from iter0 to iter9 -> iter1 to iter10
#     key_from = f"pass@1_unit_iter{i}"
#     key_to = f"pass@1_unit_iter{i+1}"
#     for sample in data:
#         from_status = "Correct" if sample.get(key_from, 0) == 1 else "Incorrect"
#         to_status = "Correct" if sample.get(key_to, 0) == 1 else "Incorrect"
#         pair = (f"{from_status} {i}", f"{to_status} {i+1}")
#         counts[pair] = counts.get(pair, 0) + 1

# # Convert to node indices
# node_labels = sorted(set([k[0] for k in counts] + [k[1] for k in counts]))
# label_to_index = {label: i for i, label in enumerate(node_labels)}

# for (src_label, tgt_label), count in counts.items():
#     sources.append(label_to_index[src_label])
#     targets.append(label_to_index[tgt_label])
#     values.append(count)

# # Plot
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=node_labels,
#     ),
#     link=dict(
#         source=sources,
#         target=targets,
#         value=values,
#     ))])

# fig.update_layout(title_text="Sankey Diagram of pass@1_unit over Iterations", font_size=10)
# fig.show()

import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data
jsonl_path = "../results/ds.jsonl"
records = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

# Build per-iteration stats
results = {"iteration": [], "Correct": [], "Incorrect": []}
for i in range(11):
    correct = sum(1 for r in records if r.get(f"pass@1_unit_iter{i}", 0) == 1)
    incorrect = len(records) - correct
    results["iteration"].append(i)
    results["Correct"].append(correct)
    results["Incorrect"].append(incorrect)

# DataFrame for plotting
df = pd.DataFrame(results)

# Define modern, soft colors
correct_color = "#66c2a5"     # soft green
incorrect_color = "#fc7d92"   # soft orange-red

# Prepare plot
plt.figure(figsize=(10, 6))
plt.stackplot(df["iteration"], df["Correct"], df["Incorrect"],
              colors=[correct_color, incorrect_color])

# Add in-plot labels
mid_iter = df["iteration"].iloc[len(df)//2]
mid_correct = df["Correct"].iloc[len(df)//2] // 2
mid_incorrect = df["Correct"].iloc[len(df)//2] + df["Incorrect"].iloc[len(df)//2] // 2

plt.text(mid_iter, mid_correct, "Passed", color="black", fontsize=12, ha="center", va="center")
plt.text(mid_iter, mid_incorrect, "Failed", color="black", fontsize=12, ha="center", va="center")

plt.xlabel("Iteration")
plt.ylabel("Number of Samples")
plt.title("pass@1_unit status over Iterations")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ramp_stacked_area.png", dpi=300)
plt.show()


