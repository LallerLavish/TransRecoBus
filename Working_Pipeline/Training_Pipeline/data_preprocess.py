import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences 
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

class Training:
    def __init__(self):
        self.train("Data/Dataset","Data/image_labels.csv")

    @staticmethod
    def preprocess_image(img_input, img_size=(1024, 256), training=False, show=False):
        if isinstance(img_input, str):  # If input is a path
            img_path = os.path.join('Data/Dataset', img_input)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img_input, np.ndarray):  # If input is an image array
            img = img_input
        else:
            raise ValueError("Input must be either a path (str) or an image array (np.ndarray)")
        
        show_image(img, 'Preprocessed Image1 (OpenCV)', show)
        img = cv2.resize(img, img_size)
        show_image(img, 'Preprocessed Image2 (OpenCV)', show)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        show_image(img, 'Preprocessed Image3 (OpenCV)', show)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
        img_tensor = tf.expand_dims(img_tensor, axis=-1)
        if training:
            img_tensor = tf.image.random_flip_left_right(img_tensor)
            img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
            img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
        img_tensor_np = img_tensor.numpy()
        img_tensor_np = np.squeeze(img_tensor_np, axis=-1)
        return img_tensor

    @staticmethod
    def preprocess_image_batch(image_inputs, training=False, img_size=(1024, 256), show=False):
        return tf.stack([Training.preprocess_image(p, img_size, training, show) for p in image_inputs])

    @staticmethod
    def preprocess_label(file_path):
        data = pd.read_csv(file_path)
        label = list(data['label'])
        vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()'\"-<>/|\=_&@$`%*}[]{ "
        char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        char_to_idx['<blank>'] = len(char_to_idx)  # Reserve the last index for <blank>
        char_to_idx['<unk>'] = len(char_to_idx) + 1
        label_matrix = []
        max_len = 2048  # Update max_len to accommodate wider paragraphs

        for i in range(len(label)):
            label_internal = []
            for ch in label[i]:
                idx = char_to_idx.get(ch, char_to_idx['<unk>'])
                label_internal.append(idx)
            label_matrix.append(label_internal)

        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        padded_matrix = pad_sequences(label_matrix, maxlen=max_len, padding='post', truncating='post')

        with open("char_to_idx.json", "w") as f:
            json.dump(char_to_idx, f)
        with open("idx_to_char.json", "w") as f:
            json.dump(idx_to_char, f)
        return padded_matrix, idx_to_char, char_to_idx

    def train(self,file_path_images,file_path_label):
        img_list=sorted(os.listdir(file_path_images))
        labels, _, char_to_idx = self.preprocess_label(file_path_label)
        label = np.where(labels >= len(char_to_idx), 0, labels)  # Ensure label indices are valid
        assert not np.any(np.isnan(labels)), "Label contains NaNs"
        assert not np.any(np.isinf(labels)), "Label contains Infs"
        input = self.preprocess_image_batch(img_list)
        input = input.astype('float32') / 255.0  # Normalize input
        assert not np.any(np.isnan(input)), "Input contains NaNs"
        assert not np.any(np.isinf(input)), "Input contains Infs"
        with open('labels.npy','wb') as file:
            np.save(file,labels)

        with open('input.npy','wb') as file:
            np.save(file,input)

        return labels,input

obj=Training()
