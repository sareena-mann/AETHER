import os
import pandas as pd

# Set the base directory containing subfolders like "yes", "no", etc.
base_dir = "/Users/japjot/aether/AETHER/model/datasets"  # change this to your directory path

# Valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Collect image paths and labels
data = []

for label in os.listdir(base_dir):
    label_path = os.path.join(base_dir, label)
    if os.path.isdir(label_path):
        for filename in os.listdir(label_path):
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(label, filename)  # relative path
                data.append({"filename": full_path, "label": label})

# Create a DataFrame and export to CSV
df = pd.DataFrame(data)
df.to_csv("gesture_dataset.csv", index=False)

print("CSV created with", len(df), "entries.")