# Using Keras to load our model
from keras.models import load_model

# To remove TF warnings & have system sleep while VideoCapture is booting
import os
import time

# To increase dimensions of np arrays
import numpy as np

# To capture frames from web camera and detect faces from frames
import cv2

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to dataset, to model, and to save/load history of model
MODEL_PATH = 'face_mask_model'

# Target size our model was trained on
TARGET_SIZE = (150, 150)

# Constants used for displaying cv2 frame String & colour respective of whether the model predicts Wearing/Not wearing
MASK_STR = ["Wearing Mask", "Not Wearing Mask"]
MASK_COL = [(0, 255, 0), (0, 0, 255)]

# Load in our previously trained mask model
mask_model = load_model(MODEL_PATH)

# Load in the haarcascade algorithm classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Grabs video capture of default web camera
vs = cv2.VideoCapture(0)
# Waiting while video feed is initializing
print("Initializing video feed please wait...")
time.sleep(5.0)


def predict_image_array(image_array):
    # Rescales image to what we tested our model on and adds additional level of nesting to array
    image_rescale = image_array / 255.0
    image_batch = np.expand_dims(image_rescale, axis=0)

    # Gets prediction of passed image and converts it to a binary 1 or 0 depending on model prediction
    predict = (mask_model.predict(image_batch) > 0.5).astype("int32")

    # True if wearing a mask - False if not
    return predict[0][0]


while True:
    # Capture the current frame and disregard the return value
    _, frame = vs.read()

    # Converting frame to gray scale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass gray scale frame to the haarcascade algorithm  with a 30% reduction in face size & a relatively higher
    # minNeighbors parameter to reduce false positive as currently this is more intended for a solo individual use.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)

    # Converting frame from BGR to RGB
    colour = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction_image = cv2.resize(colour, TARGET_SIZE)

    prediction = predict_image_array(prediction_image)

    # Iterate over all faces captured by the face detect CNN
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces and colour the rectangle according to our prediction
        cv2.rectangle(frame, (x, y), (x + w, y + h), MASK_COL[prediction], cv2.LINE_4)
        # Put text above the rectangle with offset of 20px to the right and 10px higher than rectangle &
        # place the respective coloured string according to our prediction of person is wearing a mask.
        cv2.putText(frame, MASK_STR[prediction], (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, MASK_COL[prediction], 2)

    # Display the resulting frame with the provided title
    cv2.imshow("Face Mask Detector", frame)

    # Wait 1ms between frame captures
    key = cv2.waitKey(1)

    # if the key `q` or escape was pressed, break from the loop
    if key == ord('q') or key == 27:
        break

# Closing all windows - clean up
cv2.destroyAllWindows()