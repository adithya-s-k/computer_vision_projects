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

def calculate_angle(a,b,c):#shoulder, elbow, wrist
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle 
def calculate_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    print(a)
    print(b)
    
    #distance = ((((b[0] - a[0])**(2)) - ((b[1] - a[1])**(2)))**(0.5))
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    
    return distance
def curl_counter(goal_curls):
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
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # Calculate angle
                angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                
                
                # Get coordinates of right hand
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                
                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow_l, [640, 480]).astype(int)), 
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
                cv2.putText(image, 'Do Left arm next', (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter_r > counter:
                cv2.putText(image, 'Do Right arm next', (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter == inputGoal and counter_r == inputGoal:
                cv2.putText(image, 'GOOD JOB', (540,960-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
                
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
                
            cv2.imshow('CURL COUNTER', image)
            
            
            #Tells is you are standing in range
            '''if int(shoulder_r[0]) > 0.7:
                cv2.putText(image, 'MOVE RIGHT', (100,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0,0,0), 2, cv2.LINE_AA)
                print("MOVE RIGHT")
            if int(shoulder_l[0]) < 0.3:
                cv2.putText(image, 'MOVE LEFT', (100,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0,0,0), 2, cv2.LINE_AA)
                print("MOVE LEFT")'''
    
            if int(counter) >= int(inputGoal) and int(counter_r) >= int(inputGoal):
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows() 
def push_up_counter(goal_push):
    inputGoal = goal_push
    #initializing variables to count repetitions
    counter_l=0
    counter_r=0
    stage_=None
    stage_r=None  
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
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x , landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                foot_l = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                body_angle = calculate_angle(shoulder_l,foot_l,wrist_l)
                back_angle = calculate_angle(shoulder_l,hip_l,foot_l)

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
                    counter_l +=1
                    print("Left : ",counter_l)

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
            cv2.putText(image, str(counter_l), (1280-220+10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Stage data
            cv2.putText(image, 'STAGE', (1280-220+80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage_, (1280-220+80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            
            if (back_angle + back_angle_r)/2 <= 150:
            #for posture instrustions
                cv2.rectangle(image, (0,830), (600,897), (0,0,0), -1)
                cv2.putText(image, 'Straighten your back', (15,880), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            

            #for the instructor
            cv2.rectangle(image, (0,900), (1280,960), (0,0,0), -1)
            if counter_l < counter_r:
                cv2.putText(image, 'pushup uneven, please exert force from your left hand', (15,940), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
            elif counter_r > counter_l:
                cv2.putText(image, 'pushup uneven, please exert force from your right hand', (15,940), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
            elif counter_l == inputGoal and counter_r == inputGoal:
                cv2.putText(image, 'GOOD JOB!!', (540,900), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
            # Render detections
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            cv2.imshow('Mediapipe Feed', image)
            if int(counter_l) >= int(inputGoal) and int(counter_r) >= int(inputGoal):
                time.sleep(5)
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
def squat_counter(goal_squat):
    inputGoal = goal_squat
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
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # Calculate angle
                angle = calculate_angle(hip_l, knee_l, ankle_l)
                
                
                # Get coordinates of right hand
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                # Calculate angle
                angle_r = calculate_angle(hip_r, knee_r, ankle_r)
                
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
            cv2.putText(image, str(counter), (1280-220+10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Stage data
            cv2.putText(image, 'STAGE', (1280-220+80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (1280-220+80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.rectangle(image, (730,960-60), (1280,960), (0,0,0), -1)
            if counter > counter_r:
                cv2.putText(image, 'Do Left arm next', (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter_r > counter:
                cv2.putText(image, 'Do Right arm next', (750,960-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('Mediapipe Feed', image)

            if int(counter) >= int(inputGoal) and int(counter_r) >= int(inputGoal):
                print("GOOD JOB")
                cv2.putText(image, 'GOOD JOB', (300,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
                break
                
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    return ["Squat Done",counter,counter_r]