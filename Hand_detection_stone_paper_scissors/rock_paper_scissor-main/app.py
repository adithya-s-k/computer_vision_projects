import cv2
import mediapipe as mp
import time
import numpy as np
import streamlit as st



class handDetector:
    def __init__(
        self,
        mode=False,
        maxHands=2,
        detectionCon=0.5,
        trackCon=0.5,
        csv_path="dataset.csv",
    ):
        self.static_image_mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.static_image_mode, self.maxHands, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        # Gesture recognition model
        self.csv_path = csv_path
        self.file = np.genfromtxt(self.csv_path, delimiter=",")
        self.angle = self.file[:, :-1].astype(np.float32)
        self.label = self.file[:, -1].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.angle, cv2.ml.ROW_SAMPLE, self.label)

    def findHands(self, img, draw=True):
        rps_gesture = {0: "rock", 5: "paper", 9: "scissors"}
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks is not None:
            rps_result = []
            for res in self.results.multi_hand_landmarks:
                if draw:
                    cv2.rectangle(img, (0, 0), (640, 80), (0, 255, 0), cv2.FILLED)
                    self.mpDraw.draw_landmarks(img, res, self.mpHands.HAND_CONNECTIONS)
                    # self.mpDraw.draw_landmarks(img, res,
                    #                             self.mpHands.HAND_CONNECTIONS,self.mpDraw.DrawingSpec(color=(16,218,232), thickness=2, circle_radius=4),
                    #                     self.mpDraw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                # Get angle using arcos of dot product
                angle = np.arccos(
                    np.einsum(
                        "nt,nt->n",
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                    )
                )  # [15,]
                angle = np.degrees(angle)  # Convert radian to degree
                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                # print(data)
                ret, results, neighbours, dist = self.knn.findNearest(data, 5)
                idx = int(results[0][0])
                if idx in rps_gesture.keys():
                    org = (
                        int(res.landmark[0].x * img.shape[1]),
                        int(res.landmark[0].y * img.shape[0]),
                    )
                    cv2.putText(
                        img,
                        text=rps_gesture[idx].upper(),
                        org=(org[0], org[1] + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                    )

                    rps_result.append({"rps": rps_gesture[idx], "org": org})
                if len(rps_result) >= 2:
                    winner = None
                    text = ""

                    if rps_result[0]["rps"] == "rock":
                        if rps_result[1]["rps"] == "rock":
                            text = "Tie"
                        elif rps_result[1]["rps"] == "paper":
                            text = "Paper wins"
                            winner = 1
                        elif rps_result[1]["rps"] == "scissors":
                            text = "Rock wins"
                            winner = 0
                    elif rps_result[0]["rps"] == "paper":
                        if rps_result[1]["rps"] == "rock":
                            text = "Paper wins"
                            winner = 0
                        elif rps_result[1]["rps"] == "paper":
                            text = "Tie"
                        elif rps_result[1]["rps"] == "scissors":
                            text = "Scissors wins"
                            winner = 1
                    elif rps_result[0]["rps"] == "scissors":
                        if rps_result[1]["rps"] == "rock":
                            text = "Rock wins"
                            winner = 1
                        elif rps_result[1]["rps"] == "paper":
                            text = "Scissors wins"
                            winner = 0
                        elif rps_result[1]["rps"] == "scissors":
                            text = "Tie"

                    if winner is not None:
                        cv2.putText(
                            img,
                            text="Winner",
                            org=(
                                rps_result[winner]["org"][0],
                                rps_result[winner]["org"][1] + 70,
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 255, 0),
                            thickness=3,
                        )

                    cv2.putText(
                        img,
                        text=text,
                        org=(150, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 0, 255),
                        thickness=3,
                    )

        try:
            print(data)
        except:
            data = None

        return img, data


def main():
    pTime = 0
    cTime = 0
    wCam, hCam = 640, 480
    # streamlit start
    st.title("Rock Paper scissor")
    detectionConfidence = st.slider("Hand Detection Confidence")
    trackConfidence = st.slider("Hand Tracking Confidence")
    flip_the_video = st.selectbox("Horizontally flip video ",("Yes","No"))
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    # streamlit end

    detector = handDetector(
        detectionCon=detectionConfidence / 100,
        trackCon=trackConfidence / 100,
        csv_path="dataset.csv",
    )
    if run:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                continue
            if flip_the_video =="Yes":
                img = cv2.flip(img, 1)
            elif flip_the_video == "No":
                pass
            
            img, data = detector.findHands(img, draw=True)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(
                img,
                str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3,
            )
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        else:
            st.write("Stopped")


if __name__ == "__main__":
    main()

# streamlit run app.py
