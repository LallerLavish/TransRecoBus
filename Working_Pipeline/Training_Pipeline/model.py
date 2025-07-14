import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv2D, MaxPooling2D, Bidirectional, Input, Reshape, Permute
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import sys
import os

def load_numpy(file_path):
    with open(file_path,'rb') as file:
         return np.load(file)

input = load_numpy("Resources/input.npy")


input = input.reshape((-1, 256, 1024, 1)).astype(np.float32)
label = load_numpy("Resources/labels.npy")

max_len = 500
import json

with open("Resources/char_to_idx.json", "r") as f:
    char_to_idx = json.load(f)

from tensorflow.keras.layers import Activation, Dropout, LayerNormalization,BatchNormalization

model = Sequential()

model.add(Input(shape=(256, 1024, 1)))

# Block 1
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))  # downsample height only
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

# Final pooling on width to reduce time steps from 1024 to 256
model.add(MaxPooling2D(pool_size=(1, 4)))

# Prepare for recurrent layers
model.add(Permute((2, 1, 3)))
model.add(Reshape((256, -1)))

# Normalize before RNN
model.add(LayerNormalization())

# BiGRU layers
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(Bidirectional(GRU(256, return_sequences=True)))

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