import cv2

# Load SIFT detector
sift = cv2.SIFT_create()

# Load and preprocess your crown template
crown_template = cv2.imread(r"C:\Users\thoma\Desktop\python_work\Mini_projects\DAKI2_Mini_Project_AI_systemer\Crown background removed.png", cv2.IMREAD_GRAYSCALE)
kp_template, desc_template = sift.detectAndCompute(crown_template, None)

def detect_crowns_with_sift(tile_bgr, match_threshold=10, debug=False):
    gray_tile = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    kp_tile, desc_tile = sift.detectAndCompute(gray_tile, None)

    if desc_tile is None:
        return 0

    # Match features using Brute Force Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_template, desc_tile, k=2)

    # Apply Lowe's ratio test to find good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if debug:
        matched_image = cv2.drawMatches(crown_template, kp_template, gray_tile, kp_tile, good_matches, None, flags=2)
        cv2.imshow("SIFT Matches", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If there are enough good matches, we consider it a crown
    return 1 if len(good_matches) >= match_threshold else 0