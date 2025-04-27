import pandas as pd
import numpy as np

GROUND_TRUTH_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Ground truth, creating and splitting\Ground truth with crown count.csv"
TRAIN_FILE = "ground_truth_train.csv"
TEST_FILE = "ground_truth_test.csv"
TRAIN_SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_STATE = 42  # to make the split reproducible

# Load full ground truth
df = pd.read_csv(GROUND_TRUTH_FILE)

# Sort by board and tile position (for neatness)
df = df.sort_values(by=["image_id", "y", "x"]).reset_index(drop=True)

# Find unique boards (image IDs)
unique_boards = df["image_id"].unique()

# Shuffle the boards randomly
np.random.seed(RANDOM_STATE)  # But reproducible through this
np.random.shuffle(unique_boards)

# Split boards into train/test
train_cutoff = int(len(unique_boards) * TRAIN_SPLIT_RATIO)
train_boards = unique_boards[:train_cutoff]
test_boards = unique_boards[train_cutoff:]

# Filter the full dataframe
train_df = df[df["image_id"].isin(train_boards)]
test_df = df[df["image_id"].isin(test_boards)]

# Save to separate files
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"Split complete: {len(train_boards)} training boards, {len(test_boards)} testing boards.")
print(f"Training data saved as '{TRAIN_FILE}'")
print(f"Testing data saved as '{TEST_FILE}'")