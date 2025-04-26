import os
import cv2
import pandas as pd

BOARD_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards"
OUTPUT_FILE = "tile_gt_with_crowns.csv"
TILE_SIZE = 100  # Size of each tile

# Function to Show Resized Tile
def show_tile_window(title, tile, scale=3):
    
    resized_tile = cv2.resize(
        tile, 
        (tile.shape[1] * scale, tile.shape[0] * scale), 
        interpolation=cv2.INTER_NEAREST
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, resized_tile)
    cv2.waitKey(1)

# Load Existing Progress (if available)
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    done_images = set(existing_df["image_id"])
    all_data = existing_df.to_dict('records')
else:
    done_images = set()
    all_data = []

# Main Loop
for filename in sorted(os.listdir(BOARD_FOLDER)):
    if not filename.lower().endswith(".jpg"):
        continue

    image_id = int(filename.split('.')[0])
    if image_id in done_images:
        continue

    image_path = os.path.join(BOARD_FOLDER, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {filename}")
        continue

    h, w = image.shape[:2]

    print(f"\n Labeling tiles for board: {filename} (Image ID: {image_id})")
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                continue  # Skip incomplete tiles

            title = f"Tile at ({x}, {y})"
            show_tile_window(title, tile)

            while True:
                label = input(f"Enter label for tile at (x={x}, y={y}) in {filename} (e.g. 'Forest 0'): ").strip()
                if label:
                    break
                print("Label cannot be empty. Try again.")

            all_data.append({
                "image_id": image_id,
                "x": x,
                "y": y,
                "label": label
            })

            cv2.destroyWindow(title)

    # Save after each image
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved progress after image {filename}")
    cv2.destroyAllWindows()

print("\n Tile-level ground truthing complete.")