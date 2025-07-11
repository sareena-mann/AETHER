#POV this is a cheat code, mediapipe does everything for us...how hard do i want to make this for myself

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Camera device index --> Video Capture Object
device = 0
source = cv2.VideoCapture(device)

win_name = 'Laptop Camera Input'
#WINDOW_NORMAL: resizable by user, change to WINDOW_AUTOSIZE?
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#Instantiate mp Hands model
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cv2.waitKey(1) != 27: # Escape key
        has_frame, frame = source.read()
        if not has_frame:
            break

        #Detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)