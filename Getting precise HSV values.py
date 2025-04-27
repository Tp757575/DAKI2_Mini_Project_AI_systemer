import cv2
import numpy as np

# --- Config ---
IMAGE_PATH = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards\56.jpg"  # <-- Change to any board you want to sample

# --- Load Image ---
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError("Could not load the board image.")

# Convert to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# --- Mouse Callback Function ---
def show_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = image_hsv[y, x]
        print(f"Clicked at ({x},{y}) -> HSV: {hsv_value}")

# --- Setup OpenCV Window ---
cv2.namedWindow("Board")
cv2.setMouseCallback("Board", show_hsv_value)

print("Click on tiles to print HSV values (Press ESC to quit)")

while True:
    cv2.imshow("Board", image_bgr)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
