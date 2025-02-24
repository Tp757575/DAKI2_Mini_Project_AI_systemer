import cv2
import numpy as np
import os
import random
import shutil
import pandas as pd

# Load the perspective-fixed board image
image_path = "C:\\Users\\thoma\\Desktop\\python_work\\Mini_projects\\DAKI2_Mini_Project_AI_systemer\\King Domino dataset\\Cropped and perspective corrected boards\\1.jpg"
image = cv2.imread(image_path)

# Define the grid size (King Domino uses a 5x5 grid)
GRID_SIZE = 5
TILE_SIZE = image.shape[0] // GRID_SIZE  # Assuming the image is square

# Define tile color ranges (in HSV format) for classification
color_ranges = {
    "Grass": ((35, 40, 40), (85, 255, 255)),        # Green
    "Wheat Field": ((20, 100, 100), (30, 255, 255)), # Yellow
    "Water": ((90, 50, 50), (130, 255, 255)),       # Blue
    "Swamp": ((40, 20, 20), (70, 150, 150)),       # Dark greenish
    "Mine": ((0, 0, 0), (50, 50, 50)),             # Dark gray/black
    "Forest": ((25, 50, 30), (40, 255, 100)),      # Brownish green
    "Desert": ((10, 100, 100), (20, 255, 255)),    # Light brown
}

# Convert image to HSV for color detection
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create directories for training and test data
train_dir = "training_data"
test_dir = "test_data"

# Remove existing folders and recreate
for folder in [train_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Prepare tile dataset
tile_data = []
tile_map = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        # Extract tile region
        x_start, y_start = col * TILE_SIZE, row * TILE_SIZE
        x_end, y_end = x_start + TILE_SIZE, y_start + TILE_SIZE
        tile = hsv_image[y_start:y_end, x_start:x_end]

        # Determine tile type by checking color
        detected_type = "Unknown"
        for tile_type, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(tile, np.array(lower), np.array(upper))
            if np.any(mask):  # If any pixels match the color range
                detected_type = tile_type
                break
        
        # Store result in tile map
        tile_map[row, col] = detected_type

        # Save the tile image for training/test dataset
        tile_img = image[y_start:y_end, x_start:x_end]
        tile_filename = f"tile_{row}_{col}.jpg"
        tile_data.append((tile_img, detected_type, tile_filename))

        # Draw tile boundary and label
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)
        cv2.putText(image, detected_type[:3], (x_start + 10, y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Shuffle the data randomly before splitting
random.shuffle(tile_data)

# Split into 80% training and 20% testing
split_index = int(len(tile_data) * 0.8)
train_data = tile_data[:split_index]
test_data = tile_data[split_index:]

# Save images into respective folders and record labels
train_labels = []
test_labels = []

for dataset, folder, label_list in [(train_data, train_dir, train_labels), (test_data, test_dir, test_labels)]:
    for tile_img, tile_type, filename in dataset:
        img_path = os.path.join(folder, filename)
        cv2.imwrite(img_path, tile_img)
        label_list.append((filename, tile_type))

# Save labels to CSV files for ML training
train_labels_df = pd.DataFrame(train_labels, columns=["filename", "tile_type"])
test_labels_df = pd.DataFrame(test_labels, columns=["filename", "tile_type"])

train_labels_df.to_csv(os.path.join(train_dir, "labels.csv"), index=False)
test_labels_df.to_csv(os.path.join(test_dir, "labels.csv"), index=False)

# Show the detected grid with labels
cv2.imshow("Detected Tiles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the detected tile map for debugging
print("\nTile Map:")
for row in tile_map:
    print(" | ".join(row))

print(f"\nTraining data saved in '{train_dir}' ({len(train_data)} samples).")
print(f"Test data saved in '{test_dir}' ({len(test_data)} samples).")
print("Labels saved as CSV files for future ML training.")
