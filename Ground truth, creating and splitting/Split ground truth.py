import pandas as pd
import numpy as np

# Define paths and filenames
GROUND_TRUTH_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Ground truth, creating and splitting\Ground truth with crown count.csv"
TRAIN_FILE = "ground_truth_train.csv"
TEST_FILE = "ground_truth_test.csv"

# Define how much of the data should be used for training
TRAIN_SPLIT_RATIO = 0.8  # 80% training, 20% testing
RANDOM_STATE = 42  # Setting a random seed to make the split reproducible

# Load the full ground truth dataset
df = pd.read_csv(GROUND_TRUTH_FILE)

# Sort the dataset by board and tile position so it looks cleaner and more organized
df = df.sort_values(by=["image_id", "y", "x"]).reset_index(drop=True)

# Find all unique boards based on image_id
unique_boards = df["image_id"].unique()

# Shuffle the list of boards randomly, but keep it reproducible with a fixed random seed
np.random.seed(RANDOM_STATE)
np.random.shuffle(unique_boards)

# Calculate the split point for training boards
train_cutoff = int(len(unique_boards) * TRAIN_SPLIT_RATIO)

# Separate boards into a training set and a testing set
train_boards = unique_boards[:train_cutoff]
test_boards = unique_boards[train_cutoff:]

# Filter the main dataframe into two parts based on board IDs
train_df = df[df["image_id"].isin(train_boards)]
test_df = df[df["image_id"].isin(test_boards)]

# Save the training and testing datasets into separate CSV files
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# Print summary information so we know the split was successful
print(f"Split complete: {len(train_boards)} training boards, {len(test_boards)} testing boards.")
print(f"Training data saved as '{TRAIN_FILE}'")
print(f"Testing data saved as '{TEST_FILE}'")