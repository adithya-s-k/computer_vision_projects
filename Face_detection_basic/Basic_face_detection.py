import cv2
import mediapipe as mp
import numpy as np
import time as time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,0,0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,245,0), thickness=2, circle_radius=2)
                                 )               
        
        if results.pose_landmarks != None:
            print("Start ")

            counter = 0    
            for i in range(0,11):
                if results.pose_landmarks.landmark[i].x >= 0 and results.pose_landmarks.landmark[i].x <= 1 and results.pose_landmarks.landmark[i].y >= 0 and results.pose_landmarks.landmark[i].y <= 1:
                    counter += 0
                else:
                    counter += 1
            if counter >= 1:
                print("Complete face not deteceted")
            elif counter >= 8:
                print("No face detected") 
            else:
                print("Complete face detected")
                            
            print("End ----- \n")
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()