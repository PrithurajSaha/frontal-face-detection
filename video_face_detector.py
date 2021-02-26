import cv2
from random import randrange

# load preloaded data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# webcam
webcam = cv2.VideoCapture(0)

# frame
while True:

    # read frame
    succesful_frame_read, frame = webcam.read()

    # convert Grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # draw rectangle around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    # load video
    cv2.imshow('face detector', frame)
    key = cv2.waitKey(1)

    # quit when q presses
    if key == 81 or key == 113:
        break

# Relaese video capture
webcam.release()

print('Code Completed')