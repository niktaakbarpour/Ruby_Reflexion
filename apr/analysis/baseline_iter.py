import matplotlib.pyplot as plt
import pandas as pd

# This script plots the Pass@1 performance across 12 iterations for LANTERN, ChatRepair, and RAMP, using distinct colors and markers, and saves the chart as baseline_iter.png.
# Data
data = {
    "iteration": list(range(12)),
    "LANTERN": [11.76, 26.47, 35.29, 41.17, 55.88, 58.82, 61.7, 61.7, 61.7, 61.7, 61.7, 61.7],
    "ChatRepair": [11.76, 14.7, 14.7, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6],
    "RAMP": [55, 58, 58, 61, 64, 67, 67, 67, 67, 67, 67, 67]
}

df = pd.DataFrame(data)

# Define hex colors
colors = {
    "LANTERN": "#cb4335",
    "ChatRepair": "#418a49",
    "RAMP": "#3c5c8f"        # green
}

# Plot all in one chart
plt.figure(figsize=(7, 5))
plt.plot(df["iteration"], df["LANTERN"], marker="o", linewidth=2, label="LANTERN", color=colors["LANTERN"])
plt.plot(df["iteration"], df["ChatRepair"], marker="s", linewidth=2, label="ChatRepair", color=colors["ChatRepair"])
plt.plot(df["iteration"], df["RAMP"], marker="^", linewidth=2, label="RAMP", color=colors["RAMP"])

# plt.title("Pass@1 Performance Across Iterations")
plt.xlabel("Iteration")
plt.ylabel("Pass@1 (%)")
plt.ylim(0, 100)  # consistent scale
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("baseline_iter.png", format='png')
plt.show()
