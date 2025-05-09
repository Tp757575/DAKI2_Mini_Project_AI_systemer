import pandas as pd
import numpy as np

# Define the input and output file paths
GROUND_TRUTH_TILE_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Ground truth, creating and splitting\ground_truth_train.csv"
OUTPUT_BOARD_LEVEL_FILE = "ground_truth_train_board_scores.csv"
#GROUND_TRUTH_TILE_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Ground truth, creating and splitting\ground_truth_test.csv"
#OUTPUT_BOARD_LEVEL_FILE = "ground_truth_test_board_scores.csv"

# Set the size of the board grid (5x5 tiles in this case)
GRID_SIZE = 5

# Load the tile-level ground truth dataset
tile_df = pd.read_csv(GROUND_TRUTH_TILE_FILE)

# Helper function to split a label into terrain type and crown count
# For example, "Forest 1" becomes ("Forest", 1)
def parse_label(label):
    parts = label.rsplit(' ', 1)
    if len(parts) == 2:
        terrain, crowns = parts[0], int(parts[1])
    else:
        terrain, crowns = label, 0
    return terrain, crowns

# This list will store the computed board-level scores
board_scores = []

# Group the tiles by their board (image_id) and calculate the score for each board
for image_id, group in tile_df.groupby("image_id"):
    # Initialize empty maps to store the terrain type and crown count for each tile
    tile_map = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    crown_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Fill the tile_map and crown_map with data from the CSV
    for _, row in group.iterrows():
        x, y, label = row["x"], row["y"], row["label"]
        col = x // 100  # Each tile is assumed to be 100x100 pixels
        row_idx = y // 100
        terrain, crowns = parse_label(label)
        tile_map[row_idx, col] = terrain
        crown_map[row_idx, col] = crowns

    # Prepare to calculate the score by marking all tiles as unvisited
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    total_score = 0

    # Depth-First Search function to find connected regions of the same terrain type
    def dfs(r, c, terrain_type):
        if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
            return 0, 0
        if visited[r, c] or tile_map[r, c] != terrain_type:
            return 0, 0
        visited[r, c] = True
        tiles, crowns = 1, crown_map[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            t, c_ = dfs(r + dr, c + dc, terrain_type)
            tiles += t
            crowns += c_
        return tiles, crowns

    # Go through each tile and calculate the total board score
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if not visited[i, j] and tile_map[i, j] is not None:
                size, crowns = dfs(i, j, tile_map[i, j])
                if crowns > 0:
                    total_score += size * crowns

    # Save the computed score along with the board ID
    board_scores.append({"image_id": image_id, "ground_truth_score": total_score})

# Save all board scores into a new CSV file
board_df = pd.DataFrame(board_scores)
board_df.to_csv(OUTPUT_BOARD_LEVEL_FILE, index=False)

print(f"Board-level ground truth file saved as '{OUTPUT_BOARD_LEVEL_FILE}'")