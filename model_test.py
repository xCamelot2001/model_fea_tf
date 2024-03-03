import cv2
import numpy as np
from keras.models import load_model

# Load the emotion detection model
# Make sure the model name matches the one you've saved
model = load_model('emotion_detection_model_v1.h5')
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the face cascade
# Update the path if needed to where your haarcascade_frontalface_default.xml is located
face_cascade = cv2.CascadeClassifier('/Users/camelot/model_fea_tf/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

# To capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face and predict emotion
    for (x, y, w, h) in faces:
        # Extract face region from grayscale image
        face_region = gray[y:y+h, x:x+w]

        # Resize face region to the input size expected by the model
        face_region_resized = cv2.resize(face_region, (IMG_WIDTH, IMG_HEIGHT))
    
        # Normalize pixel values if your model expects normalization
        face_region_resized = face_region_resized / 255.0

        # Expand dimensions to fit model input shape
        face_region_resized = np.expand_dims(face_region_resized, axis=0)
        face_region_resized = np.expand_dims(face_region_resized, axis=3)

        # Predict emotion
        pred = model.predict(face_region_resized)
        emotion = CLASS_LABELS[np.argmax(pred)]

        # Draw rectangle around the face and put label
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display
    cv2.imshow('Emotion Detection', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
