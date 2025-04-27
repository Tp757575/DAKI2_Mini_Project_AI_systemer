import cv2
import numpy as np

# Define the path to the image that we want to analyze
IMAGE_PATH = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards\56.jpg"  # Change this if you want to sample from a different board

# Load the image in BGR color space (which is the default in OpenCV)
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError("Could not load the board image.")

# Convert the loaded image to HSV color space because it is easier to work with colors in HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Define a function that will be called whenever we click on the image
# This function prints the HSV value at the clicked location
def show_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = image_hsv[y, x]
        print(f"Clicked at ({x},{y}) -> HSV: {hsv_value}")

# Create a window and set up the mouse click callback function
cv2.namedWindow("Board")
cv2.setMouseCallback("Board", show_hsv_value)

print("Click on tiles to print HSV values (Press ESC to quit)")

# Display the image and wait for user input
# The loop keeps running until the ESC key is pressed
while True:
    cv2.imshow("Board", image_bgr)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key pressed
        break

# After pressing ESC, close all OpenCV windows
cv2.destroyAllWindows()