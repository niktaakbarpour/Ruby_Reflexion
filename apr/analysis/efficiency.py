import matplotlib.pyplot as plt

# This script plots a scatter chart of pass@1 accuracy (%) versus runtime (seconds) for different methods (e.g., RAMP, ChatRepair, LANTERN), using a log-scaled x-axis and labeled annotations for each method.
# (name, time_seconds, pass_at_1_percent)
data = [
    ("RAMP",              23862,   67.0),
    ("RAMP E.S.",              18392,   67.0),
    ("ChatRepair",        15339,   17.6),
    ("Self-Planning",      8577,   56),
    ("Self-Collaboration", 23507,    0.0),
    ("Few-Shot",          22235,   47.5),
    ("Zero-Shot",          1727,   24.1),
    ("LANTERN",          529692,   61.7),
]

# Sort by time to keep annotations tidy (optional)
data = sorted(data, key=lambda x: x[1])

times = [t for _, t, _ in data]
passes = [p for _, _, p in data]

fig, ax = plt.subplots(figsize=(8, 5))

# Scatter points
ax.scatter(times, passes, s=300, c="#117a65")

# Annotate each dot with method name and pass@1
for name, t, p in data:
    if name == "RAMP":
        ax.annotate(f"{name}\n{p:g}%", (t, p),
            textcoords="offset points", xytext=(20, 0), ha="left", fontsize=12)
    elif name == "RAMP E.S.":
        ax.annotate(f"{name}\n{p:g}%", (t, p),
            textcoords="offset points", xytext=(-60, 1), ha="left", fontsize=12)
    elif name == "LANTERN":
        ax.annotate(f"{name}\n{p:g}%", (t, p),
            textcoords="offset points", xytext=(-40, 10), ha="left", fontsize=12)
    elif name == "Zero-Shot":
        ax.annotate(f"{name}\n{p:g}%", (t, p),
            textcoords="offset points", xytext=(-10, 10), ha="left", fontsize=12)
    else:
        ax.annotate(f"{name}\n{p:g}%", (t, p),
                    textcoords="offset points", xytext=(-23, 10), ha="left", fontsize=12)

# Axes labels and title
ax.set_xlabel("Time (s)")
ax.set_ylabel("pass@1 (%)")
# ax.set_title("pass@1 vs. Time")

# Use log scale for time (toggle if desired)
USE_LOG_X = True
if USE_LOG_X:
    ax.set_xscale("log")

# Y limits for some breathing room
ymax = max(passes)
ax.set_ylim(0, max(75, ymax + 5))

# Light grid
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
# Save if you want a file:
plt.savefig("pass1_vs_time.png", dpi=300, bbox_inches="tight")
plt.show()
