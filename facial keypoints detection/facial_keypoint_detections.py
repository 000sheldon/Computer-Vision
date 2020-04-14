from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# STEP ONE: Load dataset and preprocessing
ROOT = "C:\\Users\\liu68/Documents\\computer_vision_tf2\\dataset\\facial_keypoints_detection"
train_data = pd.read_csv(os.path.join(ROOT, 'training.csv'))
test_data = pd.read_csv(os.path.join(ROOT, 'test.csv'))

# train_data = train_data.dropna(axis=0)
# because train_data has too many nan value, we can't drop the row containing nan value.
# I fill the nan value with the forward value
train_data = train_data.fillna(method='ffill', inplace=False)

# split images and keypoints, and convert the image data to correct dtype
IMAGE_WIDTH, IMAGE_HEIGHT = 96,  96

train_x = train_data['Image']
train_images = []
for i in range(len(train_x)):
    img = np.array(train_x[i].split(' '), dtype=np.float).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    train_images.append(img)
train_images = np.array(train_images).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)


train_y = train_data.drop('Image', axis=1)
train_keypoints = []
for i in range(len(train_y)):
    keypoint = np.array(train_y.iloc[i, :])
    train_keypoints.append(keypoint)
train_keypoints = np.array(train_keypoints)

# split
train_images, val_images, train_keypoints, val_keypoints = model_selection.train_test_split(
    train_images, train_keypoints, test_size=0.2
)


test_id = test_data.drop("Image", axis=1)
test_x = test_data['Image']
test_images = []
for i in range(len(test_x)):
    img = np.array(test_x[i].split(' '), dtype=np.float).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    test_images.append(img)
test_images = np.array(test_images)

# STEP TWO: build model
model = Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(96, 96, 1)),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPool2D(pool_size=(2, 2)),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.ReLU(),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.1),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.ReLU(),
    layers.MaxPool2D(pool_size=(2, 2)),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.ReLU(),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.1),

    layers.Dense(30)
])

model.summary()

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae']) # Mean Absolute Error


# STEP THREE: training model
EPOCHS = 15
history = model.fit(train_images, train_keypoints,
                    batch_size=64, epochs=EPOCHS, validation_data=(val_images, val_keypoints))


# STEP FOUR: evaluate model

# plot loss and mae curve
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

fig, ax = plt.subplots(2, 1)
ax[0].plot(range(EPOCHS), loss, label='train_mse_loss')
ax[0].plot(range(EPOCHS), val_loss, label='val_mes_loss')
ax[0].legend()

ax[1].plot(range(EPOCHS), mae, label='train_mae')
ax[1].plot(range(EPOCHS), val_mae, label='val_mae')
ax[1].legend()
plt.savefig('mse and mae.png')

test_keypoints = model.predict(test_images)

# show one of the prediction
def visualization(image, keypoint):
    fig, ax = plt.subplots()
    image = image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
    for i in range(0, len(keypoint), 2):
        if 0 <= keypoint[i] < IMAGE_WIDTH and 0 <= keypoint[i+1] < IMAGE_HEIGHT:
            image[int(keypoint[i+1]), int(keypoint[i])] = 255

    ax.imshow(image)
    plt.savefig(f'predict_keypoints_sample')

visualization(test_images[0], test_keypoints[0])



























