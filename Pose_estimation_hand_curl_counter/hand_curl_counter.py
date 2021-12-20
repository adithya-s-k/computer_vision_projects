import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

inputGoal = int(input("Enter your rep goal for each arm: "))

cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,960)

def calculate_angle(a,b,c):
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
            
            # Curl counter logic for left
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                print("Left : ",counter)

            # Curl counter logic for right
            if angle_r > 160:
                stage_r = "down"
            if angle_r < 30 and stage_r =='down':
                stage_r="up"
                counter_r +=1
                print("Right : ",counter_r)                       
        
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (100,50), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (10,10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_r), 
                    (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (40,10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_r, 
                    (40,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        
        # Render curl counter for left hand
        # Setup status box for left hand
        
        cv2.rectangle(image, (500,0), (600,50), (245,117,0), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (10+500,10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10+500,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (40+500,10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (40+500,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
             
        cv2.imshow('Mediapipe Feed', image)

        if counter + counter_r == 2*inputGoal:
            print("GOOD JOB")
            time.sleep(5)
            break
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 