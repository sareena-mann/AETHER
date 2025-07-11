#Epoch 1: get this to work with Mediapipe then just use np to make our own NN

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
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cv2.waitKey(1) != 27: # Escape key
        has_frame, frame = source.read()
        if not has_frame:
            break

        #Detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #results.multi_hand_landmarks gives x, y, z coordinates array of [axis: value]
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                #HAND_CONNECTIONS: [part of hand (i.e. wrist): another part]
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(win_name, image)

source.release()
cv2.destroyWindow(win_name)