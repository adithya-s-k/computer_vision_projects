import cv2
cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()

    cv2.imshow("Webcam",img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
