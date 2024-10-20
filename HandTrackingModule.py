import cv2
import mediapipe as mp
import time


class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils  # To draw landmarks

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Check if any hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                # Draw the hand landmarks on the image
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # handLms.landmark is a list of all the landmarks
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # To convert these normalized coordinates to actual pixel coordinates
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                # (cx, cy) is the center of the circle.
                if draw:
                    cv2.circle(img, (cx, cy), 25,
                               (255, 0, 255), cv2.FILLED)
        return lmList
