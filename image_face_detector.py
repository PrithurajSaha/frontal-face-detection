import cv2
from random import randrange

# load preloaded data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load img
# img = cv2.imread('anne-marie.jpg')
# img = cv2.imread('prs2.jpg')
# img = cv2.imread('faces.jpg')
# img = cv2.imread('faces2.jpg')
img = cv2.imread('faces3.jpg')


# greyscale_image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# Detect face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangle around face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)
# print(face_coordinates)

# load img in window with faces
cv2.imshow('face detector', img)
cv2.waitKey()


print('Code Completed')


