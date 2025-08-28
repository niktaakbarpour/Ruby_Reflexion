# # # import json
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # # Load data
# # # with open('C:/Users/niktakbr/Desktop/Ruby_Reflexion/results/first_edgeIO_CoTIO_CoT.jsonl', 'r', encoding='utf-8') as f:
# # #     data = [json.loads(line) for line in f]

# # # df = pd.DataFrame(data)

# # # # Define difficulty bins
# # # bins = [0, 1200, 1400, 1600, 1800, 2000, 3000]
# # # labels = ['<1200', '1200–1400', '1400–1600', '1600–1800', '1800–2000', '2000+']
# # # df['difficulty_bin'] = pd.cut(df['difficulty'], bins=bins, labels=labels, right=False)

# # # # Count solved vs unsolved per bin
# # # bar_data = df.groupby(['difficulty_bin', 'is_solved']).size().unstack(fill_value=0)

# # # # Choose hex colors for bars: one for 'False' (unsolved), one for 'True' (solved)
# # # custom_colors = ['#cb4335', '#117a65']  # red-orange for False, greenish-teal for True

# # # # Plot grouped bar chart with custom colors
# # # ax = bar_data.plot(kind='bar', figsize=(10, 6), color=custom_colors)

# # # plt.xlabel('Difficulty Range')
# # # plt.ylabel('Number of Problems')
# # # plt.legend(title='is_solved', labels=['False', 'True'])
# # # plt.xticks(rotation=45)
# # # plt.tight_layout()
# # # plt.savefig("difficulty.png", format='png')
# # # plt.show()

# # import json
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # --- CONFIGURATION ---
# # jsonl_path = "../results/ds.jsonl"  # Change to your actual file
# # difficulty_min = 750
# # difficulty_max = 2500
# # bin_width = 250

# # # --- READ AND TRANSFORM DATA ---
# # verdict_data = []

# # with open(jsonl_path, 'r', encoding='utf-8') as f:
# #     for line in f:
# #         item = json.loads(line)
# #         difficulty = item.get("difficulty", None)
# #         if difficulty is None or not (difficulty_min <= difficulty <= difficulty_max):
# #             continue

# #         test_results = item.get("final_unit_test_results", [])
# #         for test in test_results:
# #             verdict = test.get("verdict", "UNKNOWN")
# #             verdict_data.append({
# #                 "difficulty": difficulty,
# #                 "verdict": verdict
# #             })

# # # --- BUILD DATAFRAME ---
# # df = pd.DataFrame(verdict_data)

# # # Bin difficulty into intervals
# # bins = pd.interval_range(start=difficulty_min, end=difficulty_max + bin_width, freq=bin_width, closed="left")
# # df["difficulty_bin"] = pd.cut(df["difficulty"], bins=bins)

# # # Count verdicts per difficulty bin
# # grouped = df.groupby(["difficulty_bin", "verdict"]).size().unstack(fill_value=0)

# # # Ensure all known verdicts are included as columns
# # all_verdicts = ["SUCCESS", "RUNTIME ERROR", "TIME_LIMIT_EXCEEDED", "COMPILATION_ERROR", "MEMORY_LIMIT_EXCEEDED", "UNKNOWN"]
# # grouped = grouped.reindex(columns=all_verdicts, fill_value=0)

# # # --- PLOT ---
# # plt.figure(figsize=(12, 6))
# # grouped.plot(kind="bar", stacked=True, colormap="tab20", width=0.9)

# # plt.xlabel("Difficulty Level")
# # plt.ylabel("Number of Test Cases")
# # plt.title("Distribution of Test Case Verdicts by Difficulty Level")
# # plt.xticks(rotation=45)
# # plt.grid(axis='y', linestyle=':', alpha=0.7)
# # plt.tight_layout()
# # plt.legend(title="Test Verdict", loc="upper right")
# # # plt.savefig("verdict_distribution_by_difficulty.png", dpi=300)
# # plt.show()


import json

# Path to your JSONL file
jsonl_path = "../results/ds_fixed.jsonl"  # Change this if needed

unique_verdicts = set()

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        test_results = obj.get("final_unit_test_results", [])
        for test in test_results:
            verdict = test.get("verdict", "UNKNOWN")
            unique_verdicts.add(verdict)

print("Unique verdicts found in dataset:")
for v in sorted(unique_verdicts):
    print("-", v)

# import json

# input_path = "../results/ds.jsonl"          # CHANGE to your input file
# output_path = "../results/ds_fixed.jsonl"   # Output path

# def should_fix(test):
#     return (
#         test.get("actual", "") == "" and
#         test.get("verdict") == "RUNTIME ERROR" and
#         not test.get("passed", False) and
#         any(e.strip() != "" for e in test.get("expected", []))
#     )

# with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         entry = json.loads(line)

#         test_results = entry.get("final_unit_test_results", [])
#         for test in test_results:
#             if should_fix(test):
#                 test["verdict"] = "WRONG ANSWER"
#                 test["info"] = "Empty actual output with non-empty expected output"
        
#         # Write the updated entry
#         json.dump(entry, outfile)
#         outfile.write("\n")

# print(f"Fixed file written to: {output_path}")

