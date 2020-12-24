import keras
from keras.preprocessing import image
from keras import models
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Variables
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

train_datagen = image.ImageDataGenerator(rotation_range=60,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         rescale=1.0/255,
                                         validation_split=0.2)

test_datagen = image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory('./data',
                                                    target_size=(150, 150),
                                                    class_mode='binary',
                                                    color_mode="rgb",
                                                    subset="training")

valid_generator = train_datagen.flow_from_directory('./data',
                                                    target_size=(150, 150),
                                                    class_mode='binary',
                                                    color_mode="rgb",
                                                    subset="validation")

images, labels = next(train_generator)


# Utility function that prints all images from a batch of the training images augmenter
def plot_images(images_array):
    fig, axes = plt.subplots(1, batch_size, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_array, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_images(images)
print(labels)

model = models.Sequential([Conv2D(filters=32,
                                  kernel_size=3,
                                  activation="relu",
                                  padding="same"),
                           MaxPool2D(pool_size=2,
                                     strides=2),
                           Conv2D(filters=64,
                                  kernel_size=3,
                                  activation="relu",
                                  padding="same"),
                           MaxPool2D(pool_size=2,
                                     strides=2),
                           Conv2D(filters=64,
                                  kernel_size=3,
                                  activation="relu",
                                  padding="same"),
                           MaxPool2D(pool_size=2,
                                     strides=2),
                           Flatten(),
                           Dense(units=64,
                                 activation="relu"),
                           Dense(units=1,
                                 activation="sigmoid")])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="binary_crossentropy",
              metrics=["accuracy"])


history = model.fit(train_generator, epochs=EPOCHS, validation_data=valid_generator, batch_size=BATCH_SIZE)