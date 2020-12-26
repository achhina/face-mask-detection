# Using Keras to load our model and images
from keras.models import load_model

# To grab environment variables, image directories, and image paths
import os
import time

# To turn our lists into numpy arrays
import numpy as np

import cv2

# Stops TF optimization warnings from displaying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to dataset, to model, and to save/load history of model
MODEL_PATH = 'face_mask_model'

model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vs = cv2.VideoCapture(0)
time.sleep(5.0)


def predict_image_array(image_array):
    # Convert to a numpy array and adds additional level of nesting to array
    image_rescale = image_array / 255.0
    image_batch = np.expand_dims(image_rescale, axis=0)

    # Gets prediction of passed image
    prediction = (model.predict(image_batch) > 0.5).astype("int32")
    # True if wearing a mask - False if not
    return prediction[0][0] == 0.0


while True:
    # Capture frame-by-frame
    ret, frame = vs.read()

    # Converting frame to gray scale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Converting frame from BGR to RGB
    colour = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction_image = cv2.resize(colour, (150, 150))

    boolean = predict_image_array(prediction_image)
    string = "Wearing Mask"
    if not boolean:
        string = "Not Wearing Mask"
    box = (0, 255, 0) if boolean else (0, 0, 255)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), box, cv2.LINE_4)
        cv2.putText(frame, string, (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box, thickness=2)

    # Display the resulting frame
    cv2.imshow("Face Mask Detector", frame)

    # Wait 1ms between frame captures
    key = cv2.waitKey(1)

    # if the key `q` or escape was pressed, break from the loop
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()