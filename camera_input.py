import cv2
import sys

#Camera device index
device = 0

#Checks if there is a command line specification to override device values
if len(sys.argv) > 1:
    device = sys.argv[1]

#Video Capture object
source = cv2.VideoCapture(device)

win_name = 'Laptop Camera Input'
#WINDOW_NORMAL: resizable by user, change to WINDOW_AUTOSIZE?
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape key
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)