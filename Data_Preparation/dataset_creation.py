import cv2
import pytesseract
import numpy as np
import csv
import matplotlib.pyplot as plt
def extract_printed_label(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray,cmap='gray')
    plt.show()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    plt.imshow(blurred,cmap='gray')
    plt.show()
    # Binarize image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    plt.imshow(binary)
    plt.show()


    # Morphological operation to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_positions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100:  # filter short lines
            line_positions.append(y)

    if len(line_positions) < 2:
        raise ValueError("Less than two horizontal lines found")

    # Sort lines by vertical position
    line_positions.sort()
    y_top = line_positions[0]
    y_bottom = line_positions[1]

    cropped = gray[y_top:y_bottom, :]
    _, thresh = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binary, cmap='gray')
    plt.show()
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text.strip()

def clean_label(label):
    lines = label.splitlines()
    cleaned_lines = [line.lstrip(": ").rstrip() for line in lines]
    return " ".join(cleaned_lines)

import os

folder_path = "Dataset"
files = sorted(os.listdir(folder_path))
image_label_pairs = []

for f in files:
    full_path = os.path.join(folder_path, f)
    if os.path.isfile(full_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            label = clean_label(extract_printed_label(full_path))
            image_label_pairs.append([full_path, label])
        except Exception as e:
            print(f"Error processing {f}: {e}")

# Save to CSV
csv_path = "image_labels.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])
    writer.writerows(image_label_pairs)

print(f"Dataset saved to {csv_path}")