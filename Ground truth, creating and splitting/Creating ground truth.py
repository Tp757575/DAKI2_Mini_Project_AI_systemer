import os
import cv2
import pandas as pd

# Define the folder where the board images are stored
BOARD_FOLDER = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards"

# Define the name of the output CSV file that will store the manual labels
OUTPUT_FILE = "tile_gt_with_crowns.csv"

# Define the size of each tile (assuming 100x100 pixel tiles)
TILE_SIZE = 100

# This function shows a tile in a resizable window and scales it up for easier viewing
def show_tile_window(title, tile, scale=3):
    resized_tile = cv2.resize(
        tile, 
        (tile.shape[1] * scale, tile.shape[0] * scale), 
        interpolation=cv2.INTER_NEAREST
    )
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, resized_tile)
    cv2.waitKey(1)

# Check if a labeling session was already started earlier
# If yes, load the existing labeled data so we don't lose previous progress
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    done_images = set(existing_df["image_id"])
    all_data = existing_df.to_dict('records')
else:
    done_images = set()
    all_data = []

# Go through all images in the board folder
for filename in sorted(os.listdir(BOARD_FOLDER)):
    # Only process JPEG files
    if not filename.lower().endswith(".jpg"):
        continue

    # Extract image ID (based on filename) and skip if already labeled
    image_id = int(filename.split('.')[0])
    if image_id in done_images:
        continue

    # Load the full board image
    image_path = os.path.join(BOARD_FOLDER, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {filename}")
        continue

    h, w = image.shape[:2]

    print(f"\n Labeling tiles for board: {filename} (Image ID: {image_id})")
    
    # Divide the board into tiles and go through each one
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            # Skip incomplete tiles at the edges
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                continue

            # Show the current tile enlarged for easier manual inspection
            title = f"Tile at ({x}, {y})"
            show_tile_window(title, tile)

            # Ask the user to manually label the tile (terrain type and crown count)
            while True:
                label = input(f"Enter label for tile at (x={x}, y={y}) in {filename} (e.g. 'Forest 0'): ").strip()
                if label:
                    break
                print("Label cannot be empty. Try again.")

            # Save the manual label for this tile
            all_data.append({
                "image_id": image_id,
                "x": x,
                "y": y,
                "label": label
            })

            # Close the current tile window after labeling
            cv2.destroyWindow(title)

    # Save all progress to the output file after finishing each full board
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved progress after image {filename}")
    cv2.destroyAllWindows()

print("\n Tile-level ground truthing complete.")