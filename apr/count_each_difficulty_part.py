import json
import matplotlib.pyplot as plt
from collections import Counter

# Path to your JSONL file
file_path = "benchmarks/merged_output.jsonl"

# Initialize a dictionary to hold the counts of outcomes by difficulty
outcome_counts_by_difficulty = {
    "COMPILATION_ERROR": Counter(),
    "RUNTIME_ERROR": Counter(),
    "MEMORY_LIMIT_EXCEEDED": Counter(),
    "TIME_LIMIT_EXCEEDED": Counter(),
    "WRONG_ANSWER": Counter()
}

# Read the JSONL file and populate the counts
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            difficulty = entry.get("difficulty")
            bug_outcome = entry.get("bug_exec_outcome")
            if difficulty and bug_outcome in outcome_counts_by_difficulty:
                outcome_counts_by_difficulty[bug_outcome][difficulty] += 1
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

# Prepare the data for plotting
# Get difficulties from the first outcome (assuming all outcomes have the same set of difficulties)
difficulties = list(outcome_counts_by_difficulty["COMPILATION_ERROR"].keys())

outcome_data = {outcome: [outcome_counts_by_difficulty[outcome].get(difficulty, 0) for difficulty in difficulties]
                for outcome in outcome_counts_by_difficulty}

# Set color mapping for the outcomes (your specified colors)
colors = {
    "COMPILATION_ERROR": "#cecece",
    "RUNTIME_ERROR": "#f0c571",
    "MEMORY_LIMIT_EXCEEDED": "#e02b35",
    "TIME_LIMIT_EXCEEDED": "#a559aa",
    "WRONG_ANSWER": "#59a89c"
}

# Create a stacked bar chart
plt.figure(figsize=(10, 6))

bar_width = 40  # Bar width
bottom = [0] * len(difficulties)  # Start with a base of 0 for stacking

for outcome, color in colors.items():
    plt.bar(difficulties, outcome_data[outcome], bar_width, bottom=bottom, label=outcome, color=color)
    bottom = [bottom[i] + outcome_data[outcome][i] for i in range(len(difficulties))]  # Update the base for the next stack

# Adding labels and title
plt.xlabel("Difficulty Level")
plt.ylabel("Number of Entries")
plt.title("Distribution of Bug Execution Outcomes by Difficulty Level")
plt.xticks(rotation=45)
plt.legend(title="Bug Execution Outcome")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
