import cv2
import numpy as np

# Path to your board image
IMAGE_PATH = r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\King Domino dataset\Cropped and perspective corrected boards\34.jpg"

# Function to normalize brightness
def normalize_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v))
    normalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return normalized

# Load and normalize the image
image_bgr = cv2.imread(IMAGE_PATH)
image_bgr = normalize_image(image_bgr)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Click callback to print HSV
def show_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = image_hsv[y, x]
        print(f"Clicked at ({x},{y}) -> HSV: {hsv_value}")

# Setup window
cv2.namedWindow("Normalized Image")
cv2.setMouseCallback("Normalized Image", show_hsv_value)

while True:
    if cv2.getWindowProperty("Normalized Image", cv2.WND_PROP_VISIBLE) < 1:
        break
    cv2.imshow("Normalized Image", image_bgr)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()