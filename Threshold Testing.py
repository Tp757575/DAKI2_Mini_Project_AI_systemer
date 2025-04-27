import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up constants and paths used throughout the code
GRID_SIZE = 5
TILE_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards"
CROWN_TEMPLATE_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Cropped crowns"
GROUND_TRUTH_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\ground_truth_train_board_scores.csv"

# Define HSV color ranges for detecting different terrain types
COLOR_RANGES = {
    "Forest": ((35, 30, 20), (55, 255, 120)),
    "Field": ((15, 240, 50), (30, 255, 190)),
    "Lake": ((105, 200, 50), (115, 255, 160)),
    "Mine": ((0, 30, 15), (30, 255, 200)),
    "Grassland": ((33, 50, 120), (45, 255, 160)),
    "Swamp": ((2, 70, 35), (22, 220, 130))
}

# Define the HSV range that should capture crown colors
CROWN_HSV_RANGE = ((10, 30, 140), (30, 255, 255))

# This function loads all the crown template images from a folder
def load_crown_templates(folder_path):
    crown_templates = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.png'):
            template_path = os.path.join(folder_path, file)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                crown_templates.append(template)
    return crown_templates

# Load crown templates once so they can be reused during detection
crown_templates = load_crown_templates(CROWN_TEMPLATE_FOLDER)

# Load a full board image based on its filename
def load_board_image(filename):
    return cv2.imread(os.path.join(TILE_FOLDER, filename))

# Try to classify a tile by checking which HSV color range it matches
def classify_tile_color(tile_hsv):
    for tile_type, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(tile_hsv, np.array(lower), np.array(upper))
        if np.any(mask):
            return tile_type
    return "Unknown"

# Detect whether there is a crown inside a given tile
def detect_crowns_in_tile(tile_bgr, templates, threshold):
    tile_hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    lower, upper = CROWN_HSV_RANGE
    crown_mask = cv2.inRange(tile_hsv, np.array(lower), np.array(upper))
    yellow_pixels = cv2.countNonZero(crown_mask)
    if yellow_pixels < 10:
        return 0

    gray_tile = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray_tile = cv2.GaussianBlur(gray_tile, (3, 3), 0)

    best_match_score = 0
    for template in templates:
        res = cv2.matchTemplate(gray_tile, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        best_match_score = max(best_match_score, max_val)

    return 1 if best_match_score >= threshold else 0

# Build maps of the terrain types and crown counts for a full board
def build_tile_and_crown_maps(image, templates, threshold):
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
            crowns = detect_crowns_in_tile(tile_bgr, templates, threshold)

            tile_map[row, col] = tile_type
            crown_map[row, col] = crowns

    return tile_map, crown_map

# Calculate the total score for a board based on connected regions of the same terrain
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

# Test different template matching thresholds and calculate the average error for each
thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
average_errors = []

gt_df = pd.read_csv(GROUND_TRUTH_FILE)

for threshold in thresholds:
    total_error = 0
    count = 0

    for _, row in gt_df.iterrows():
        image_id = row["image_id"]
        filename = f"{image_id}.jpg"
        actual_score = row["ground_truth_score"]

        image = load_board_image(filename)
        tile_map, crown_map = build_tile_and_crown_maps(image, crown_templates, threshold)
        predicted_score = calculate_score(tile_map, crown_map)

        total_error += abs(predicted_score - actual_score)
        count += 1

    average_error = total_error / count
    average_errors.append(average_error)
    print(f"Threshold {threshold:.2f} --> Average Error: {average_error:.2f}")

# Plot the average errors for each threshold value
plt.figure(figsize=(8, 6))
plt.plot(thresholds, average_errors, marker='o')
plt.title("Average Error vs Matching Threshold")
plt.xlabel("Template Matching Threshold")
plt.ylabel("Average Error (points)")
plt.grid(True)
plt.show()
