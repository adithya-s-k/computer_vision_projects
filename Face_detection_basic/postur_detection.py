import cv2
import math
import mediapipe as mp
import numpy as np
import time as time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    print(a)
    print(b)
    
    #distance = ((((b[0] - a[0])**(2)) - ((b[1] - a[1])**(2)))**(0.5))
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    
    return distance

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
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                   
            # Calculate angle
            distance_cal = (calculate_distance(nose,shoulder)*100)
            print(distance_cal)
        
        except:
            pass
        
        cv2.rectangle(image, (0,0), (70,80), (245,117,16), -1)
        cv2.putText(image, str(distance_cal), (10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
           
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,0,0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,245,0), thickness=2, circle_radius=2)
                                 )               
        

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()