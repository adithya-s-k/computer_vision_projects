import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)

# define a video capture object
vid = cv2.VideoCapture(0)
TIMER = int(10)
while TIMER > 0:
    time.sleep(1)
    TIMER -= 1
    print(TIMER)


while(True):
    

    ret, frame = vid.read()
  
    cv2.putText(frame, 'TIME', (560,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(TIMER), (560,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()