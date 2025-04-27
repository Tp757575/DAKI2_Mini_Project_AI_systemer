import os
import cv2
import numpy as np
import pandas as pd

# Define constants for board size and file paths
GRID_SIZE = 5
TILE_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards"
CROWN_TEMPLATE_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Cropped crowns"
GROUND_TRUTH_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\ground_truth_train_board_scores.csv"

# Define HSV color ranges for different terrain types
COLOR_RANGES = {
    "Forest": ((35, 30, 20), (55, 255, 120)),
    "Field": ((15, 240, 50), (30, 255, 190)),
    "Lake": ((105, 200, 50), (115, 255, 160)),
    "Mine": ((0, 30, 15), (30, 255, 200)),
    "Grassland": ((33, 50, 120), (45, 255, 160)),
    "Swamp": ((2, 70, 35), (22, 220, 130))
}

# Define the HSV color range expected for crowns
CROWN_HSV_RANGE = ((10, 30, 140), (30, 255, 255))

# Function to load all crown template images into a list
def load_crown_templates(folder_path):
    crown_templates = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.png'):
            template_path = os.path.join(folder_path, file)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                crown_templates.append(template)
    print(f"Loaded {len(crown_templates)} crown templates from '{folder_path}'.")
    return crown_templates

# Load crown templates once at the start
crown_templates = load_crown_templates(CROWN_TEMPLATE_FOLDER)

# Load a full board image based on the filename
def load_board_image(filename):
    return cv2.imread(os.path.join(TILE_FOLDER, filename))

# Classify a tile by checking which HSV color range it falls into
def classify_tile_color(tile_hsv):
    for tile_type, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(tile_hsv, np.array(lower), np.array(upper))
        if np.any(mask):
            return tile_type
    return "Unknown"

# Detect if a crown is present in a tile using HSV pre-filtering and template matching
def detect_crowns_in_tile(tile_bgr, debug=False):
    tile_hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    lower, upper = CROWN_HSV_RANGE
    crown_mask = cv2.inRange(tile_hsv, np.array(lower), np.array(upper))

    yellow_pixels = cv2.countNonZero(crown_mask)
    if yellow_pixels < 10:
        if debug:
            print("No crown-like color detected.")
        return 0

    gray_tile = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray_tile = cv2.GaussianBlur(gray_tile, (3, 3), 0)

    best_match_score = 0
    for template in crown_templates:
        res = cv2.matchTemplate(gray_tile, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        best_match_score = max(best_match_score, max_val)

    if debug:
        print(f"Best match score: {best_match_score:.2f}")

    threshold = 0.9  # Based on experiments, this threshold gave the best balance
    return 1 if best_match_score >= threshold else 0

# Build maps that show what terrain and how many crowns are at each tile
def build_tile_and_crown_maps(image):
    tile_map = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    crown_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    tile_size = image.shape[0] // GRID_SIZE

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x_start, y_start = col * tile_size, row * tile_size
            x_end, y_end = x_start + tile_size, y_start + tile_size
            tile_bgr = image[y_start:y_end, x_start:x_end]
            tile_hsv = hsv_image[y_start:y_end, x_start:x_end]

            tile_type = classify_tile_color(tile_hsv)
            crowns = detect_crowns_in_tile(tile_bgr)

            tile_map[row, col] = tile_type
            crown_map[row, col] = crowns

    return tile_map, crown_map

# Calculate the total board score based on connected terrain regions and crown counts
def calculate_score(tile_map, crown_map):
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    total_score = 0

    def dfs(r, c, tile_type):
        if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
            return 0, 0
        if visited[r, c] or tile_map[r, c] != tile_type:
            return 0, 0
        visited[r, c] = True
        tiles, crowns = 1, crown_map[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            t, c_ = dfs(r + dr, c + dc, tile_type)
            tiles += t
            crowns += c_
        return tiles, crowns

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if not visited[i, j] and tile_map[i, j] != "Unknown":
                size, crowns = dfs(i, j, tile_map[i, j])
                if crowns > 0:
                    total_score += size * crowns
    return total_score

# Function to evaluate all boards against the ground truth data
def evaluate_against_ground_truth():
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)

    total_absolute_error = 0
    total_relative_accuracy = 0
    count = 0

    for _, row in gt_df.iterrows():
        image_id = row["image_id"]
        filename = f"{image_id}.jpg"
        actual_score = row["ground_truth_score"]

        image = load_board_image(filename)
        tile_map, crown_map = build_tile_and_crown_maps(image)
        predicted_score = calculate_score(tile_map, crown_map)

        error = abs(predicted_score - actual_score)
        relative_accuracy = (1 - error / actual_score) * 100 if actual_score != 0 else 0

        total_absolute_error += error
        total_relative_accuracy += relative_accuracy
        count += 1

    mean_absolute_error = total_absolute_error / count
    mean_relative_accuracy = total_relative_accuracy / count

    print(f"Mean Absolute Error (MAE): {mean_absolute_error:.2f}")
    print(f"Mean Relative Accuracy (MRA): {mean_relative_accuracy:.2f}%")

# Run the evaluation if this file is run as a script
if __name__ == "__main__":
    evaluate_against_ground_truth()