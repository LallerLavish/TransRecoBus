import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract
import math
import pandas as pd
import json
from glob import glob

def deskew(image):
    coords = np.column_stack(np.where(image < 255))
    if len(coords) < 2:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def segment_lines(image):
    def preprocess_image(image):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def find_line_segments(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 10, 5))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

        line_images = []
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 10
            line = image[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
            line_images.append(line)
        return line_images

    binary = preprocess_image(image)
    return find_line_segments(binary)

def run_ocr(image):
    config = '--psm 7'
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

def split_label_by_ocr(full_label, ocr_texts, padding_words=1):
    full_words = full_label.strip().split()
    total_words = len(full_words)
    ocr_word_counts = [len(t.strip().split()) for t in ocr_texts]
    total_predicted_words = sum(ocr_word_counts)

    if total_predicted_words == 0:
        return [''] * len(ocr_texts)

    scale = total_words / total_predicted_words
    split_sizes = [max(1, round(wc * scale)) for wc in ocr_word_counts]

    while sum(split_sizes) > total_words:
        for i in range(len(split_sizes)):
            if split_sizes[i] > 1:
                split_sizes[i] -= 1
                break
    while sum(split_sizes) < total_words:
        split_sizes[-1] += 1

    chunks = []
    start = 0
    for size in split_sizes:
        end = start + size
        padded_start = max(0, start - padding_words)
        padded_end = min(len(full_words), end + padding_words)
        chunk = ' '.join(full_words[padded_start:padded_end])
        chunks.append(chunk)
        start = end

    return chunks

def is_empty_line(image, pixel_threshold=0.01, min_component_area=10, min_components=2, min_height=10, min_width=30):
    h, w = image.shape
    if h < min_height or w < min_width:
        return True
    total_pixels = h * w
    non_white = np.sum(image < 200)
    pixel_ratio = non_white / total_pixels
    if pixel_ratio < pixel_threshold:
        return True
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid_components = [stat for stat in stats[1:] if stat[cv2.CC_STAT_AREA] > min_component_area]
    return len(valid_components) < min_components

# ===== Main Script =====
dataset_dir = "/kaggle/input/images"
image_paths = sorted(glob(os.path.join(dataset_dir, "*.png")))

csv_path = "/kaggle/input/labeldata/image_labels.csv"
df = pd.read_csv(csv_path)
label_map = {
    os.path.basename(row['image_path']): row['label']
    for _, row in df.iterrows()
}

all_image_tensors = []
all_encoded_labels = []
all_full_labels = []

image_label_pairs = []

# First pass: collect all valid (image, label) pairs
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    full_label = label_map.get(image_name, "")
    if not full_label:
        print(f"No label found for {image_name}, skipping.")
        continue
    image_label_pairs.append((image_path, full_label))
    all_full_labels.append(full_label)

# Build character dictionary
all_text = ''.join(all_full_labels)
unique_chars = sorted(set(all_text))
char_to_idx = {c: i + 1 for i, c in enumerate(unique_chars)}
char_to_idx['<pad>'] = 0
i=0
# Second pass: process images
while(i<1):
    for image_path, full_label in image_label_pairs:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
    
        lines = segment_lines(image)
        ocr_texts = []
        filtered_lines = []
        for line_img in lines:
            if is_empty_line(line_img):
                continue
            text = run_ocr(line_img)
            ocr_texts.append(text)
            filtered_lines.append(line_img)

        chunks = split_label_by_ocr(full_label, ocr_texts, padding_words=3)

        for line_img, label in zip(filtered_lines, chunks):
            resized = cv2.resize(line_img, (128, 32))
            normalized = resized.astype(np.float32) / 255.0
            all_image_tensors.append(normalized)
    
            encoded = [char_to_idx.get(c, char_to_idx['<pad>']) for c in label]
            all_encoded_labels.append(encoded)
    i=i+1


# Padding
pad_idx = char_to_idx['<pad>']
max_len = max(len(seq) for seq in all_encoded_labels)
padded_labels = [seq + [pad_idx] * (max_len - len(seq)) for seq in all_encoded_labels]

# Save
image_array = np.array(all_image_tensors)
label_array = np.array(padded_labels, dtype=np.int32)

np.save("line_images.npy", image_array)
np.save("line_labels.npy", label_array)

with open("char_to_idx.json", "w") as f:
    json.dump(char_to_idx, f)

print("Saved line_images.npy, line_labels.npy, and char_to_idx.json")