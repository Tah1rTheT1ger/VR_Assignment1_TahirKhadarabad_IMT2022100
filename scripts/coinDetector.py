import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Reads, resizes, converts to grayscale, and applies blur to the input image.
    """
    image = cv2.imread(image_path)

    # Resize for better visualization
    scale_percent = 80  # Adjust scale if needed
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)

    return image, gray, blurred

def detect_edges(blurred):
    """
    Applies adaptive thresholding and Canny edge detection.
    This helps detect low-contrast grey coins.
    """
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    
    edges = cv2.Canny(adaptive_thresh, 50, 150)
    return edges

def find_contours(edges):
    """
    Finds contours in the edge-detected image.
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # Use morphological closing to fill gaps in edges
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, closed

def filter_coins_by_shape(contours):
    """
    Filters contours based on area and circularity to eliminate non-coin round objects.
    """
    valid_coins = []
    rejected_objects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small noise
        if area < 500:
            continue

        # Calculate circularity = (4 * pi * area) / (perimeter^2)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue  # Avoid division by zero

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Consider a valid coin if circularity is close to 1 (perfect circle) and area is within range
        if 0.7 < circularity < 1.2 and 1000 < area < 100000:
            valid_coins.append(cnt)
        else:
            rejected_objects.append(cnt)

    return valid_coins, rejected_objects

def get_dominant_color(image, mask):
    """
    Extracts the dominant color from a masked region of the image.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get masked pixels
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

    # Convert to list of pixels
    pixels = masked_hsv.reshape(-1, 3)
    pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]  # Remove black (masked) pixels

    if len(pixels) == 0:
        return None  # No valid pixels found

    # Compute the mean color in HSV space
    avg_hue = np.mean(pixels[:, 0])  # Hue value

    return avg_hue

def filter_coins_by_color(image, valid_coins):
    """
    Filters out non-coin objects based on color.
    """
    accepted_coins = []
    rejected_by_color = []

    # Define valid coin color ranges (Hue values in HSV)
    silver_hue_range = (0, 100)  # Silver, light grey coins (low hue)
    gold_hue_range = (10, 40)   # Gold/Bronze coins

    for cnt in valid_coins:
        mask = np.zeros_like(image[:, :, 0])

        # Create mask for the coin
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Get dominant color hue
        avg_hue = get_dominant_color(image, mask)

        if avg_hue is None:
            continue  # Skip if no color detected

        # Check if the coin's color falls within a valid range
        if silver_hue_range[0] <= avg_hue <= silver_hue_range[1] or \
           gold_hue_range[0] <= avg_hue <= gold_hue_range[1]:
            accepted_coins.append(cnt)
        else:
            rejected_by_color.append(cnt)

    return accepted_coins, rejected_by_color

def segment_and_save_coins(image, valid_coins):
    """
    Segments each detected coin and saves it as an individual image.
    """
    for i, cnt in enumerate(valid_coins):
        mask = np.zeros_like(image[:, :, 0])

        # Draw filled contour to create a mask
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Extract the coin using the mask
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)

        # Crop the coin using bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_coin = segmented_coin[y:y+h, x:x+w]

        # Save segmented coin
        cv2.imwrite(f"coin_{i+1}.png", cropped_coin)
        cv2.imshow(f"Coin {i+1}", cropped_coin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = "../input_images/coins_shapes.jpg"  # Update path as needed
    # image_path = "../input_images/coins_circles.jpg"

    # Preprocess image
    image, gray, blurred = preprocess_image(image_path)

    # Detect edges
    edges = detect_edges(blurred)

    # Find contours
    contours, closed = find_contours(edges)

    # Filter valid coins by shape
    valid_coins, rejected_objects = filter_coins_by_shape(contours)

    # Further filter coins by color
    accepted_coins, rejected_by_color = filter_coins_by_color(image, valid_coins)

    # Draw accepted coins in green
    output = image.copy()
    cv2.drawContours(output, accepted_coins, -1, (0, 255, 0), 2)

    # Draw rejected objects (non-coins) in red
    cv2.drawContours(output, rejected_objects, -1, (0, 0, 255), 2)

    # Draw rejected by color in blue
    cv2.drawContours(output, rejected_by_color, -1, (255, 0, 0), 2)

    # Display results
    cv2.imshow("Detected Coins", output)
    cv2.imwrite("final_detected_coins.png", output)

    # Display the total count of detected coins
    num_coins = len(accepted_coins)
    print(f"Total number of valid coins detected: {num_coins}")

    # Segment and save coins
    segment_and_save_coins(image, accepted_coins)

if __name__ == "__main__":
    main()
