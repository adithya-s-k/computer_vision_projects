import cv2
import mediapipe as mp
import numpy as np
import math
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)

def calculate_angle(a,b,c):#HIP, KNEE, ANKLE
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle 

goal_curls = 10

inputGoal = goal_curls
# Curl counter variables
counter = 0 
counter_r = 0
stage = None
stage_r = None
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
            
            # Get coordinates
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            # Calculate angle of left full
            angle = calculate_angle(hip_l, knee_l, ankle_l)
            
            
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            # Calculate angle
            angle_r = calculate_angle(hip_r, knee_r, ankle_r)
            
            # Curl counter logic for left
            if angle > 160:
                stage = "Down"
            if angle < 120 and stage =='Down':
                stage="Up"
                counter +=1
                print("Left : ",counter)

            # Curl counter logic for right
            if angle_r > 160:
                stage_r = "Down"
            if angle_r < 120 and stage_r =='Down':
                stage_r="Up"
                counter_r +=1
                print("Right : ",counter_r)                       
        
        except:
            pass
        
        # Render curl counter for right hand
        # Setup status box for right hand
        cv2.rectangle(image, (0,0), (70,80), (0,0,0), -1)
        # cv2.rectangle(image, (0,35), (220,80), (245,117,16), -1)
        cv2.rectangle(image, (75,0), (220,80), (0,0,0), -1)
        # Rep data
        cv2.putText(image, 'REPS', (5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_r), (10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, stage_r, (80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        
        
        # Render curl counter for left hand
        # Setup status box for left 
        cv2.rectangle(image, (1280-220,0), (1280-150,80), (0,0,0), -1)
        # cv2.rectangle(image, (0,35), (220,80), (245,117,16), -1)
        cv2.rectangle(image, (1280-145,0), (1280,80), (0,0,0), -1)
        # Rep data
        cv2.putText(image, 'REPS', (1280-220+5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (1280-220+10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (1280-220+80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (1280-220+80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        
        #for the instructor
        cv2.rectangle(image, (730,960-60), (1280,960), (0,0,0), -1)
        if counter > counter_r:
            cv2.putText(image, 'Do Left Leg next', (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        elif counter_r > counter:
            cv2.putText(image, 'Do Right Leg next', (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        elif counter == inputGoal and counter_r == inputGoal:
            cv2.putText(image, 'GOOD JOB', (540,960-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
            
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               
            
        cv2.imshow('RUNNING COUNTER', image)

        if int(counter) >= int(inputGoal) and int(counter_r) >= int(inputGoal):
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows() 