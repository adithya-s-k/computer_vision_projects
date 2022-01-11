from typing import Counter
import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

inputGoal = int(input("Enter your rep goal for each arm: "))

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

def calculate_angle(a,b,c):#HIP, KNEE, ANKLE
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle 


# Curl counter variables
counter = 0 
counter_r = 0
stage = None
stage_r = None

## Setup mediapipe instance
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
            HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            # Calculate angle
            angle = calculate_angle(HIP, KNEE, ANKLE)
            
            
            # Get coordinates of right hand
            HIP_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            KNEE_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ANKLE_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            # Calculate angle
            angle_r = calculate_angle(HIP_r, KNEE_r, ANKLE_r)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(KNEE, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
            
            # Curl counter logic for left
            if angle > 150:
                    stage ="Up"
            if angle < 60 and stage == "Up":
                stage = "Down"
                counter += 1
                print("Left :", counter)

            # Curl counter logic for right
            if angle_r > 150:
                stage_r = "Up"
            if angle_r < 60 and stage_r =="Up":
                stage_r="Down"
                counter_r +=1
                print("Right : ",counter_r)                      
        
        except:
            pass
        
        # Render curl counter for right hand
        # Setup status box for right hand
        cv2.rectangle(image, (0,0), (140,55), (245,117,16), -1)
        # Rep data
        cv2.putText(image, 'REPS', (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_r), (10,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (60,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_r, (60,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        
        
        # Render curl counter for left hand
        # Setup status box for left hand
        cv2.rectangle(image, (500,0), (640,55), (245,117,0), -1)# Rectange properties rectangle(image, start_point, end_point, color, thickness) 
        # Rep data
        cv2.putText(image, 'REPS', (510,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)#cv2.putText(image,  text_to_show,  (20, 40),  fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,  fontScale=1,  color=(255, 255, 255))
        cv2.putText(image, str(counter), (510,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (560,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (560,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
             
        cv2.imshow('Mediapipe Feed', image)
        
        
        #Tells is you are standing in range
        '''if int(shoulder_r[0]) > 0.7:
            cv2.putText(image, 'MOVE RIGHT', (100,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0,0,0), 2, cv2.LINE_AA)
            print("MOVE RIGHT")
        if int(shoulder[0]) < 0.3:
            cv2.putText(image, 'MOVE LEFT', (100,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0,0,0), 2, cv2.LINE_AA)
            print("MOVE LEFT")'''
            
        if counter == inputGoal and counter_r == inputGoal:
            print("GOOD JOB")
            cv2.putText(image, 'GOOD JOB', (300,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
            time.sleep(5)
            break
        elif counter > counter_r:
            print("Right arm next")
        elif counter_r > counter:
            print("Left arm next")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()