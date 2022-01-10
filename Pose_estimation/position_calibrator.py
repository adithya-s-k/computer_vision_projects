# This programs tells you where to stand
# basically it tells you if you standing way to left or right and tells you to stand in a specified range

import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)

def calculate_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    return distance

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting BGR to RGB so that it becomes easier for library to read the image
        image.flags.writeable = False #this step is done to save some memoery
        # Make detection
        results = pose.process(image) #We are using the pose estimation model 
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            shoulder_ll = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            shoulder_rl = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            shoulder_l = round(float(shoulder_ll[0]),3)
            shoulder_r = round(float(shoulder_rl[0]),3)

            #print(shoulder_l,shoulder_r)
        except:
            pass
        
        w = calculate_distance(shoulder_rl,shoulder_ll)
        W = 6.0

        # Finding the Focal Length
        # d = 80
        # f = (w*d)/W
        # print("Focal Length :",f)

        f = 6.10
        d = (W * f) / w
        print("DISTANCE :",d)

        cv2.rectangle(image, (730,960-60), (1280,960), (0,0,0), -1)
        cv2.rectangle(image, (0,960-60), (550,960), (0,0,0), -1)
        cv2.putText(image, str(shoulder_l) , (20,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(shoulder_r), (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,255), 2, cv2.LINE_AA)
        # cv2.putText(image, 'GOOD JOB', (540,960-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               
        

        cv2.imshow('CALIBRATOR', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows() 
