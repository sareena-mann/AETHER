from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import p
import tensorflow as tf
import os
from PIL import Image

def load_wider_face_data(mat_file_path, img_dir, img_size=(12, 12)):
    mat_data = loadmat(mat_file_path)
    file_list = mat_data['file_list']
    face_bbx_list = mat_data['face_bbx_list']
    event_list = mat_data['event_list']

    images = []
    bboxes = []
    labels = []

    for i in range(len(file_list)):
        event_file_name = file_list[i][0]
        event_bbox = face_bbx_list[i][0]
        event_name = event_list[i][0][0]  # ex:'0--Parade'
        event_dir = os.path.join(img_dir, event_name)

        # Verify event directory exists
        if not os.path.exists(event_dir):
            print(f"Event directory not found: {event_dir}")
            continue

        for j in range(len(event_file_name)):
            img_name = event_file_name[j][0][0]
            img_path = os.path.join(event_dir, f"{img_name}.jpg")
            bbox_list = event_bbox[j][0]  # [[x1, y1, w, h]]

            img = Image.open(img_path).convert('RGB')
            orig_width, orig_height = img.size
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0

            for bbox in bbox_list:
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h

                # Scale bounding box to 12x12
                x1 = x1 * img_size[0] / orig_width
                y1 = y1 * img_size[1] / orig_height
                x2 = x2 * img_size[0] / orig_width
                y2 = y2 * img_size[1] / orig_height
                bbox_scaled = [x1, y1, x2, y2]

                images.append(img_array)
                bboxes.append(bbox_scaled)
                labels.append([0, 1])

    images = np.array(images, dtype=np.float32)
    bboxes = np.array(bboxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Reshape for PNet output: [batch, H_out, W_out, 4 or 2]
    # H_out, W_out = 3, 3  # Based on PNet architecture for 12x12 input
    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.float32)  # Shape: [N, 12, 12, 3]
    bboxes = np.array(bboxes, dtype=np.float32)  # Shape: [N, H_out, W_out, 4]
    labels = np.array(labels, dtype=np.float32)  # Shape: [N, H_out, W_out, 2]

    return images, [bboxes, labels]


def bbox_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

def cls_loss(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=[1, 2])  # Now shape = (N, 2)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

    # Training function
def train_pnet(model, images, targets, epochs=10, batch_size=32, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=[bbox_loss, cls_loss])

    if images is None or targets is None:
        print("No data to train on.")
        return None

    bboxes, labels = targets
    history = model.fit(
        images,
        [bboxes, labels],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1)
    return history

# Main execution
if __name__ == "__main__":
    mat_file_path = '/Users/sareenamann/AETHER/face/wider_face_split/wider_face_train.mat'
    img_dir = '/Users/sareenamann/AETHER/face/WIDER_train/images'
    # mat_file_path = '/Users/japjot/PycharmProjects/AETHER/face/wider_face_split/wider_face_train.mat'
    # img_dir = '/Users/japjot/PycharmProjects/AETHER/face/WIDER_train/images'
    
    images, targets = load_wider_face_data(mat_file_path, img_dir)

    if images is not None:
        pnet = p.PNet()
        pnet.construct(size=(None, 12, 12, 3))
        history = train_pnet(pnet, images, targets)

        pnet.save("pnet_model.keras")
        print("Model training complete and saved as pnet_model.keras")
    else:
        print("Failed to load data. Training aborted.")
