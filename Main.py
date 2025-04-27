import os
import cv2
import numpy as np
import pandas as pd

# --- Config ---
GRID_SIZE = 5
TILE_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards"
CROWN_TEMPLATE_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Cropped crowns"
GROUND_TRUTH_FILE = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\ground_truth_train_board_scores.csv"

# Color Ranges for Tile Classification (HSV)
COLOR_RANGES = {
    "Grassland": ((35, 40, 40), (85, 255, 255)),
    "Field": ((20, 100, 100), (30, 255, 255)),
    "Lake": ((90, 50, 50), (130, 255, 255)),
    "Swamp": ((40, 20, 20), (70, 150, 150)),
    "Mine": ((0, 0, 0), (50, 50, 50)),
    "Forest": ((25, 50, 30), (40, 255, 100)),
}

# --- Load Crown Templates ---
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

crown_templates = load_crown_templates(CROWN_TEMPLATE_FOLDER)

# --- Functions ---
def load_board_image(filename):
    return cv2.imread(os.path.join(TILE_FOLDER, filename))

def classify_tile_color(tile_hsv):
    for tile_type, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(tile_hsv, np.array(lower), np.array(upper))
        if np.any(mask):
            return tile_type
    return "Unknown"

def detect_crowns_in_tile(tile_bgr, debug=False):
    gray_tile = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    best_match_score = 0

    for template in crown_templates:
        res = cv2.matchTemplate(gray_tile, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        best_match_score = max(best_match_score, max_val)

    if debug:
        print(f"Best match score: {best_match_score:.2f}")

    threshold = 0.6  # Adjust if needed
    return 1 if best_match_score >= threshold else 0

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

def evaluate_against_ground_truth():
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)

    for _, row in gt_df.iterrows():
        image_id = row["image_id"]
        filename = f"{image_id}.jpg"
        actual_score = row["ground_truth_score"]

        image = load_board_image(filename)
        tile_map, crown_map = build_tile_and_crown_maps(image)
        predicted_score = calculate_score(tile_map, crown_map)

        print(f"{filename}: Predicted = {predicted_score}, Actual = {actual_score}, Error = {abs(predicted_score - actual_score)}")

# --- Run Evaluation ---
if __name__ == "__main__":
    evaluate_against_ground_truth()