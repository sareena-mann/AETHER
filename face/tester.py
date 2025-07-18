import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from face.p import PNet
from face.p_load_train import bbox_loss, cls_loss, load_wider_face_data

# Load model
model = tf.keras.models.load_model(
    'pnet_model.keras',
    custom_objects={'PNet': PNet, 'bbox_loss': bbox_loss, 'cls_loss': cls_loss}
)

# Load data
mat_file_path = '/Users/sareenamann/AETHER/face/wider_face_split/wider_face_train.mat'
img_dir = '/Users/sareenamann/AETHER/face/WIDER_train/images'
images, targets, img_paths = load_wider_face_data(mat_file_path, img_dir)

# Select one image
if len(images) == 0:
    print("No images loaded. Check your data paths.")
    exit()

# Use the first image and its metadata
img_array = images[100:110]  # Shape: [1, 12, 12, 3]
img_path, orig_width, orig_height, gt_bbox = img_paths[100]
gt_bboxes = [gt_bbox]  # Ground-truth bounding box in original image coordinates

# Make prediction
predictions = model.predict(img_array)
regressions, face = predictions
confidence = face[0, 0, 0, 1]  # Face probability
bbox_pred = regressions[0, 0, 0, :]  # Predicted bounding box
print("Face Confidence:", confidence)
print("Predicted Bounding Box (12x12 scale):", bbox_pred)

# Visualize result
img = Image.open(img_path).convert('RGB')
draw = ImageDraw.Draw(img)

# Draw predicted bounding box
if confidence > 0.5:
    scale_x, scale_y = orig_width / 12, orig_height / 12
    x1 = bbox_pred[0] * scale_x
    y1 = bbox_pred[1] * scale_y
    x2 = bbox_pred[2] * scale_x
    y2 = bbox_pred[3] * scale_y
    draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
    print(f"Scaled Predicted Box: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")

# Draw ground-truth bounding box
for x1, y1, x2, y2 in gt_bboxes:
    draw.rectangle((x1, y1, x2, y2), outline='green', width=2)
    print(f"Ground-Truth Box: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")

plt.imshow(img)
plt.axis('off')
plt.show()