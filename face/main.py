import cv2
import numpy as np
import os
import tensorflow as tf
from face.gausspyramid import constructPyramids, createFolder
from face.p import PNet
import time

"""
    Uses OpenCV to retrieve camera input
    Runs input through Gaussian Pyramid ->
    Runs returned pyramid through NN (as of now just PNet) ->
    Runs through NMS (todo)
    Projects bounding boxes onto captured image and onto feed
"""

def processPyramidWithPNet(pyramid, model, ogShape):
    all_boxes = []
    all_scores = []

    for i, img in enumerate(pyramid):
        scale = ogShape[0] / img.shape[0]

        # Normalize
        img = img.astype(np.uint8)
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
        score_mask = scores > 0.5  # Lowered threshold to 0.5 for debugging
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
            all_scores.append(score_val)

        all_boxes.extend(boxes)
    print(f"Pyramid level {i}: {len(boxes)} boxes detected with scores: {all_scores}")
    return all_boxes, all_scores

# Load PNet model from .h5 file
pnet_path = "/Users/sareenamann/AETHER/face/pnet_model.h5"
print(f"Checking PNet model file: {pnet_path}")
print(f"PNet model exists: {os.path.exists(pnet_path)}")
if not os.path.exists(pnet_path):
    raise FileNotFoundError(f"Error: PNet model file '{pnet_path}' does not exist")

# Load model weights manually
try:
    pnet = PNet()
    pnet.construct(size=(None, 12, 12, 3))
    pnet.build(input_shape=(None, 12, 12, 3))
    pnet.load_weights(pnet_path)
    pnet.summary()  # Print model summary to verify layers
    print("PNet model loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

# Camera device index --> Video Capture Object
device = 0
source = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)  # Use CAP_AVFOUNDATION for macOS
if not source.isOpened():
    print("Error: Could not open camera with device index 0. Trying alternative indices...")
    for device in [1, 2]:
        source = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
        if source.isOpened():
            print(f"Camera opened successfully with device index {device}.")
            break
    if not source.isOpened():
        print("Error: Could not open any camera.")
        raise RuntimeError("Camera initialization failed")

# Add initialization delay
time.sleep(1)  # Wait 1 second to ensure camera is ready

win_name = 'Laptop Camera Input'
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

# Create folder for results
folderName = './results/camera_feed_results'
createFolder(folderName)
createFolder(os.path.join(folderName, 'gaussian_pyramid'))
frame_count = 0

while True:
    has_frame, frame = source.read()
    print(f"Frame {frame_count} captured: {has_frame}")
    if not has_frame:
        print(f"Failed to capture frame {frame_count}. Retrying...")
        continue

    # Verify frame is valid
    if frame is None or frame.size == 0:
        print(f"Frame {frame_count} is invalid or empty.")
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pyramid = constructPyramids(gray_frame, f"frame_{frame_count}", folderName)
    print(f"Pyramid generated with {len(pyramid)} levels")
    boxes, scores = processPyramidWithPNet(pyramid, pnet, frame.shape[:2])

    print(f"Frame {frame_count}: {len(boxes)} boxes detected")
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = max(10, y1 - 10)
        cv2.putText(frame, f"{score:.2f}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(30)  # Increased delay to 30ms for smoother display
    if key == 27:  # Escape key
        break

    frame_count += 1

source.release()
cv2.destroyAllWindows()