import keras
from keras.preprocessing import image
import numpy as np

train_datagen = image.ImageDataGenerator(rotation_range=60,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         rescale=1.0/255,
                                         validation_split=0.2)

test_datagen = image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
