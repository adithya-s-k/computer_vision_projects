import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# inputGoal = int(input("Enter your rep goal for each arm: "))
inputGoal = 10
width_cam = 720
height_cam = 640
cap = cv2.VideoCapture(0)
cap.set(3,width_cam)
cap.set(4,height_cam)

def calculate_angle(a,b,c):#shoulder, elbow, wrist
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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            
            # Get coordinates of right hand
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            # Calculate angle
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
            # Visualize angle
            cv2.putText(image, str(angle),
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
            
            # Curl counter logic for left
            if angle > 160:
                stage = "Down"
            if angle < 30 and stage =='Down':
                stage="Up"
                counter +=1
                print("Left : ",counter)

            # Curl counter logic for right
            if angle_r > 160:
                stage_r = "Down"
            if angle_r < 30 and stage_r =='Down':
                stage_r="Up"
                counter_r +=1
                print("Right : ",counter_r)                       
        
        except:
            pass
        
        # Render curl counter for right hand
        # Setup status box for right hand
        cv2.rectangle(image, (0,0), (70,80), (245,117,16), -1)
        # cv2.rectangle(image, (0,35), (220,80), (245,117,16), -1)
        cv2.rectangle(image, (75,0), (220,80), (245,117,16), -1)
        # Rep data
        cv2.putText(image, 'REPS', (5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_r), (10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_r, (80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        
        
        # Render curl counter for left hand
        # Setup status box for left hand
        cv2.rectangle(image, (width_cam-220,0), (width_cam-150,80), (245,117,16), -1)
        # cv2.rectangle(image, (0,35), (220,80), (245,117,16), -1)
        cv2.rectangle(image, (width_cam-145,0), (width_cam,80), (245,117,16), -1)
        # Rep data
        cv2.putText(image, 'REPS', (width_cam-220+5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (width_cam-220+10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (width_cam-220+80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (width_cam-220+80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        
        #for the instructor
        cv2.rectangle(image, (730,height_cam-60), (1280,height_cam), (245,117,16), -1)
        if counter > counter_r:
            cv2.putText(image, 'Do Left arm next', (750,height_cam-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        elif counter_r > counter:
            cv2.putText(image, 'Do Right arm next', (750,height_cam-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        elif counter == inputGoal and counter_r == inputGoal:
            cv2.putText(image, 'GOOD JOB', (540,height_cam-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
            
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
            time.sleep(5)
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 