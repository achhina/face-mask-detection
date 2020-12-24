# Using Keras to process our training data and train our model
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

# To save and load model history
from joblib import dump, load

# To plot our data and get appropriate GRID_SIZE for subplots
import matplotlib.pyplot as plt
from math import ceil, sqrt

# To grab environment variables and perform checks
import os
from os.path import exists

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Sets models learning rate, number of passes of our training data (epochs) and batch size
LEARNING_RATE = 1e-4
EPOCHS = 30
BATCH_SIZE = 32

# Dynamically sets n * n grid size to plot our sample augmented images generated
GRID_SIZE = ceil(sqrt(BATCH_SIZE))

# Path to dataset, to model, and to save/load history of model
DATASET_PATH = './data'
MODEL_PATH = 'face_mask_model'
HISTORY_PATH = MODEL_PATH + '/history.joblib'

# Expands size of training data by making modifications to our dataset
train_datagen = image.ImageDataGenerator(rotation_range=60,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         rescale=1.0/255,
                                         validation_split=0.2)

# Generates batches of augmented training data from the directory of our dataset
train_generator = train_datagen.flow_from_directory(DATASET_PATH,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(150, 150),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    subset='training')

# Generates batches of augmented validation data from the directory of our dataset
valid_generator = train_datagen.flow_from_directory(DATASET_PATH,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(150, 150),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    subset='validation')

# Creates the layers for our Sequential model
#
# Layers used and their parameters:
#
# Conv2D - 2D convolution layer for our images (kernel moves in 2 directions and outputs data as 3 dimensional)
#       filters - number of output filters (number of different ways of extracting features from an input)
#       kernel_size - height and width of convolution window
#       activation - activation function to use
#                   (ReLU tend to blow up activation but help with vanishing gradient problem & opposite for Sigmoid)
#       padding - to add padding to the image if it is not fully covered by the filter
#       input_shape - if one not provided infers from x argument of Model.fit
#
# MaxPool2D - Reduces spatial size of convolution layer by taking maximum value of each cluster and reducing it to one
#       pool_size - window to take the maximum from the convolution layer
#       strides - how far the pooling window moves for each step
#
# Flatten - Removes all dimensions except 1 (turns data into 1D array)
#
# Dense - Layer uses a linear operation on output of last layer to input of next layer
#       units - hyperparameter tweaked through experimentation
#       activation - activation function to use
model = Sequential([Conv2D(filters=32,
                           kernel_size=3,
                           activation='relu',
                           padding='same'),
                   MaxPool2D(pool_size=2,
                             strides=2),
                   Conv2D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'),
                   MaxPool2D(pool_size=2,
                             strides=2),
                   Conv2D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'),
                   MaxPool2D(pool_size=2,
                             strides=2),
                   Flatten(),
                   Dense(units=64,
                         activation='relu'),
                   Dense(units=1,
                         activation='sigmoid')])

# Compile model using Adam optimizer and accuracy metric (metric to be evaluated during training/testing)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Trains and saves model if not done already
if not exists(MODEL_PATH):
    history = model.fit(train_generator, epochs=EPOCHS, validation_data=valid_generator, batch_size=BATCH_SIZE).history
    model.save(MODEL_PATH)
    dump(history, HISTORY_PATH)


# Utility function that plots all images from a batch of the training images augmenter
def plot_images(images_array, labels_array):
    # Create n * n subplots
    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(10, 10))
    axes = axes.flatten()

    # Add all images to subplots with their respective label
    for img, ax, label in zip(images_array, axes, labels_array):
        ax.imshow(img)
        ax.set_title(label)

    # Hide axis of all subplots whether we are displaying an image in them or not
    for i in range(0, GRID_SIZE * GRID_SIZE):
        axes[i].axis('off')

    # Fit to figure area and plot
    plt.tight_layout()
    plt.show()


# Utility function that plots Validation & Training Loss/Accuracy from our model
def plot_history(file_path):
    # Load history of our model from designated file path
    model_history = load(file_path)

    # Plots the Validation & Training Loss
    plt.plot(range(1, EPOCHS + 1), model_history['loss'], 'g', label='Training loss')
    plt.plot(range(1, EPOCHS + 1), model_history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and Validation loss'), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.legend()
    plt.show()

    # Plots the Validation & Training Accuracy
    plt.plot(range(1, EPOCHS + 1), model_history['accuracy'], 'g', label='Training accuracy')
    plt.plot(range(1, EPOCHS + 1), model_history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy'), plt.xlabel('Epochs'), plt.ylabel('Accuracy'), plt.legend()
    plt.show()


if __name__ == '__main__':
    # Grabs BATCH_SIZE array of images and labels from training generator
    imgs, labels = next(train_generator)

    # Plots sample training images along with their respective numerical label
    plot_images(imgs, labels)

    # Plots Validation & Training Loss/Accuracy
    plot_history(HISTORY_PATH)