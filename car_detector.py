# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import cv2

# capture frames from a video
cap = cv2.VideoCapture(0)

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.line(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img=frames, text="CAR", org=(x + int(w / 10), y + int(h / 1.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=3)
    # Display frames in a window
    cv2.imshow('DETECTOR', frames)

# Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()



#
# import cv2
# import time
#
# # Create our body classifier
# car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
#
# # Initiate video capture for video file
# cap = cv2.VideoCapture(0)
#
# # Loop once video is successfully loaded
# while cap.isOpened():
#
#     time.sleep(.05)
#     # Read first frame
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Pass frame to our car classifier
#     cars = car_classifier.detectMultiScale(gray, 1.4, 2)
#
#     # Extract bounding boxes for any bodies identified
#     for (x, y, w, h) in cars:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#         cv2.imshow('Cars', frame)
#
#     if cv2.waitKey(1) == 13:  # is the Enter Key
#         break
#
# cap.release()
# cv2.destroyAllWindows()