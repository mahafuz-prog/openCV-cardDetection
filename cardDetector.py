from libraries import *

# Read camera
cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    result = frame.copy()

    detectCard(frame, result)

    cv2.imshow('Main Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break