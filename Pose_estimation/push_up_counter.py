import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

goal = 5
# goal=int(input("enter number of pushups to be performed"))
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)

def calculate_angle(a,b,c):#shoulder, elbow, wrist
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle 
#variables to count repetitions
counter_=0
counter_r=0
stage_=None
stage_r=None
# Setup mediapipe instance
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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x , landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            body_angle = calculate_angle(shoulder,foot,wrist)
            back_angle = calculate_angle(shoulder,hip,foot)

            # Get coordinates of right hand
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            foot_r = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate angle
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            body_angle_r = calculate_angle(shoulder_r,foot_r,wrist_r)
            back_angle_r = calculate_angle(shoulder_r,hip_r,foot_r)

            # pushup counter logic for left
            if angle <= 90 and body_angle <= 40:
                stage_ = "Down"
            if angle > 90 and angle <= 180 and body_angle >=40 and stage_ =='Down':
                stage_="Up"
                counter_ +=1
                print("Left : ",counter_)

            # Curl counter logic for right
            if angle_r <= 90 and body_angle_r <= 40:
                stage_r = "Down"
            if angle_r > 90 and angle_r <= 180 and body_angle_r >= 40 and stage_r =='Down':
                stage_r="Up"
                counter_r +=1
                print("Right : ",counter_r)  


        except:
            pass

        # Render pushup counter for right hand
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
        # Setup status box for left hand
        cv2.rectangle(image, (1280-220,0), (1280-150,80), (0,0,0), -1)
        # cv2.rectangle(image, (0,35), (220,80), (245,117,16), -1)
        cv2.rectangle(image, (1280-145,0), (1280,80), (0,0,0), -1)
        # Rep data
        cv2.putText(image, 'REPS', (1280-220+5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_), (1280-220+10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (1280-220+80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, stage_, (1280-220+80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
        
        if (back_angle + back_angle_r)/2 <= 150:
        #for posture instrustions
            cv2.rectangle(image, (0,830), (600,897), (0,0,0), -1)
            cv2.putText(image, 'Straighten your back', (15,880), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
        

        #for the instructor
        cv2.rectangle(image, (0,900), (1280,960), (0,0,0), -1)
        if counter_ < counter_r:
            cv2.putText(image, 'pushup uneven, please exert force from your left hand', (15,940), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
        elif counter_r > counter_:
            cv2.putText(image, 'pushup uneven, please exert force from your right hand', (15,940), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
        elif counter_ == goal and counter_r == goal:
            cv2.putText(image, 'GOOD JOB!!', (540,900), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
         # Render detections
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
             
        cv2.imshow('Mediapipe Feed', image)
        
        if counter_ == goal and counter_r == goal:
            time.sleep(5)
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 