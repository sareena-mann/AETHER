import cv2
import numpy as np
import os
from face.gausspyramid import constructPyramids, createFolder
from face.p import PNet
"""
    Uses OpenCV to retrieve camera input
    Runs input through Guassian Pyramid -> 
    Runs returned pyramid through NN (as of now just PNet) ->
    Runs through NMS (todo)
    Projects bounding boxes on to captured image and onto feed
"""

def processPyramidWithPNet(pyramid, model, ogShape):
    all_boxes = []
    all_scores = []

    for i, img in enumerate(pyramid):
        scale = ogShape[0] / img.shape[0]

        #Normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = img_rgb / 255.0
        img_in = np.expand_dims(img_in, axis=0)

        regressions, score = model(img_in)

        # Get rid of batch dim
        regressions = regressions.numpy().squeeze()
        score = score.numpy().squeeze()
        scores = score[:, :, 1]

        stride = 2
        cell_size = 12
        score_mask = scores > 0.7
        indices = np.where(score_mask)

        boxes = []

        for y, x in zip(indices[0], indices[1]):
            reg = regressions[y, x]
            score_val = scores[y, x]

            # Map to original image coordinates
            x1 = (x * stride) * scale
            y1 = (y * stride) * scale
            x2 = (x * stride + cell_size) * scale
            y2 = (y * stride + cell_size) * scale

            # Adjust with regressions
            x1 += reg[0] * scale
            y1 += reg[1] * scale
            x2 += reg[2] * scale
            y2 += reg[3] * scale

            boxes.append([x1, y1, x2, y2])
            all_scores.append(score)

        all_boxes.extend(boxes)
    return all_boxes, all_scores

pnet = PNet()
pnet.load_weights('weights/pnet_weights.h5')

#Camera device index --> Video Capture Object
device = 0
source = cv2.VideoCapture(device)
win_name = 'Laptop Camera Input'
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

# Create folder for results
folderName = './results/camera_feed_results'
createFolder(folderName)
createFolder(os.path.join(folderName, 'gaussian_pyramid'))
frame_count = 0

while cv2.waitKey(1) != 27:  # Escape key
    has_frame, frame = source.read()
    if not has_frame:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pyramid = constructPyramids(gray_frame, f"frame_{frame_count}", folderName)
    boxes, scores = processPyramidWithPNet(pyramid, pnet, frame.shape[:2])

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2, = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # ADDED: Adjust text position to stay within bounds
        text_y = max(10, y1 - 10)
        cv2.putText(frame, f"{score:.2f}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(win_name, frame)
    frame_count += 1

source.release()
cv2.destroyAllWindows()