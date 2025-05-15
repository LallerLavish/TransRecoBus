import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv2D, MaxPooling2D, Bidirectional, Input, Reshape, Permute
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.util import load_numpy

input = load_numpy("input.npy")
print(input)

input = input.reshape((-1, 256, 1024, 1)).astype(np.float32)
label = load_numpy("labels.npy")

max_len = 500
import json

with open("char_to_idx.json", "r") as f:
    char_to_idx = json.load(f)

model = Sequential()

model.add(Input(shape=(256, 1024, 1)))  # 500 time steps, 1024 width, 1 channel

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))  # Downsample height only

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))  # Downsample height only

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))  # Downsample height only

model.add(MaxPooling2D(pool_size=(1, 4)))  # Reduce width from 1024 to 256 to reduce GRU time steps

model.add(Permute((2, 1, 3)))
model.add(Reshape((256, -1)))  # Adjusted after pooling

model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(Bidirectional(GRU(128, return_sequences=True)))

model.add(Dense(len(char_to_idx)+1, activation='softmax'))

def ctc_loss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    y_true = tf.pad(y_true, [[0, 0], [0, max_len - tf.shape(y_true)[1]]], constant_values=0)

    input_length = tf.ones([batch_size, 1], dtype='int32') * time_steps  # shape: (batch_size, 1)
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype='int32'), axis=1, keepdims=True)  # shape: (batch_size, 1)

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

model.compile(optimizer='adam', loss=ctc_loss)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(input, label, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stopping],verbose=2)

model.summary()

model.save("model.h5")