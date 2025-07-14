import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

def show_image(img, title='Image', show=False):
    if show:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

def enhance_for_segmentation(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def deskew_image(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def segment_lines_unbreakable(img):
    processed = enhance_for_segmentation(img)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])
    merged_lines = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        if h < 20:
            continue
        if merged_lines and (y - merged_lines[-1][1] - merged_lines[-1][3] < 20):
            prev = merged_lines.pop()
            top = min(prev[1], y)
            bottom = max(prev[1] + prev[3], y + h)
            new_box = (min(prev[0], x), top, max(prev[0] + prev[2], x + w) - min(prev[0], x), bottom - top)
            merged_lines.append(new_box)
        else:
            merged_lines.append(box)

    line_imgs = [img[y:y+h, x:x+w] for (x, y, w, h) in merged_lines]
    return line_imgs

class Training:
    def __init__(self):
        self.train("Data/Dataset", "Data/image_labels.csv")

    @staticmethod
    def preprocess_image(img_input, img_size=(1024, 256), training=False, show=False):
        if isinstance(img_input, str):
            img_path = os.path.join('Data/Dataset', img_input)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img_input, np.ndarray):
            img = img_input
        else:
            raise ValueError("Input must be path or ndarray")

        show_image(img, 'Original', show)
        img = cv2.resize(img, img_size)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
        img_tensor = tf.expand_dims(img_tensor, axis=-1)

        if training:
            img_tensor = tf.image.random_flip_left_right(img_tensor)
            img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
            img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)

        return img_tensor

    @staticmethod
    def preprocess_label(file_path):
        data = pd.read_csv(file_path)
        labels = list(data['label'])
        vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()'\"-<>/|\=_&@$`%*}[]{ "
        char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        char_to_idx['<blank>'] = len(char_to_idx)
        char_to_idx['<unk>'] = len(char_to_idx) + 1

        label_matrix = []
        max_len = 3000  # Handles long paragraphs

        for text in labels:
            line = [char_to_idx.get(ch, char_to_idx['<unk>']) for ch in str(text)]
            label_matrix.append(line)

        padded_matrix = tf.keras.preprocessing.sequence.pad_sequences(label_matrix, maxlen=max_len, padding='post')
        idx_to_char = {v: k for k, v in char_to_idx.items()}
        return padded_matrix, idx_to_char, char_to_idx

    def train(self, file_path_images, file_path_label):
        image_files = sorted(os.listdir(file_path_images))
        full_labels, idx_to_char, char_to_idx = self.preprocess_label(file_path_label)
        all_line_images = []
        all_line_labels = []

        data = pd.read_csv(file_path_label)
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(file_path_images, img_name)
            orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if orig_img is None:
                continue

            deskewed = deskew_image(orig_img)
            line_imgs = segment_lines_unbreakable(deskewed)

            full_text = str(data.iloc[i]['label'])
            avg_chars_per_line = max(len(full_text) // max(len(line_imgs), 1), 1)
            split_labels = [full_text[j:j+avg_chars_per_line] for j in range(0, len(full_text), avg_chars_per_line)]

            for j, line_img in enumerate(line_imgs):
                context_img = self.build_context_window(line_imgs, j)
                line_tensor = self.preprocess_image(context_img)
                all_line_images.append(line_tensor)

                if j < len(split_labels):
                    line_label = [char_to_idx.get(ch, char_to_idx['<unk>']) for ch in split_labels[j]]
                    all_line_labels.append(line_label)

        all_line_labels_padded = tf.keras.preprocessing.sequence.pad_sequences(all_line_labels, maxlen=512, padding='post')
        images_np = tf.stack(all_line_images).numpy()

        np.save('processed_images.npy', images_np)
        np.save('processed_labels.npy', all_line_labels_padded)
        with open("char_to_idx.json", "w") as f:
            json.dump(char_to_idx, f)
        with open("idx_to_char.json", "w") as f:
            json.dump(idx_to_char, f)

        return all_line_labels_padded, images_np

    def build_context_window(self, lines, index):
        blank = np.zeros_like(lines[index])
        prev = lines[index - 1] if index > 0 else blank
        curr = lines[index]
        next = lines[index + 1] if index < len(lines) - 1 else blank
        context = np.vstack([prev, curr, next])
        return context

# Run it
obj = Training()