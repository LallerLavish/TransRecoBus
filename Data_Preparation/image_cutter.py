import cv2
import os

def crop_image_using_second_line(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    # Convert to grayscale and apply GaussianBlur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarize the image (invert colors)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operation to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Find contours of the horizontal lines
    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_positions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100:  # Filter short lines
            line_positions.append(y)

    if len(line_positions) < 2:
        raise ValueError("Less than two horizontal lines found")

    # Sort the positions to get the second line
    line_positions.sort()
    y_second_from_top = line_positions[1]
    y_first_from_bottom = line_positions[-1]

    if y_second_from_top >= y_first_from_bottom:
        raise ValueError("Detected lines are not in valid order for cropping")

    # Crop the region between the two lines
    cropped_img = img[y_second_from_top+13:y_first_from_bottom, :]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_img)
    print(f"Image saved after cropping to: {output_path}")

# Example usage
folder_path = "Dataset"
output_folder = "Cropped_Images"
os.makedirs(output_folder, exist_ok=True)

files = sorted(os.listdir(folder_path))
for f in files:
    full_path = os.path.join(folder_path, f)
    if os.path.isfile(full_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_path = os.path.join(output_folder, f)
        crop_image_using_second_line(full_path, output_path)
