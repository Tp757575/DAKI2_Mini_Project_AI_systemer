import pandas as pd
import os

# List of manually verified boards and their scores
# Replace with your actual filenames and scores
ground_truth = [
    ("51.jpg", 37),
    ("52.jpg", 42),
    ("55.jpg", 37),
    ("56.jpg", 44),
    ("59.jpg", 38),
    ("60.jpg", 44),
    ("63.jpg", 38),
    ("64.jpg", 66),
]

# Convert to DataFrame
gt_df = pd.DataFrame(ground_truth, columns=["filename", "ground_truth_score"])

# Save to CSV
gt_df.to_csv("ground_truth_scores.csv", index=False)

print("Ground truth saved to 'ground_truth_scores.csv'")

print(os.path.abspath("ground_truth_scores.csv"))
