from scipy.io import loadmat
mat_data = loadmat('/Users/sareenamann/AETHER/face/wider_face_split/wider_face_test.mat')
print(mat_data.keys())
for key in mat_data:
    if not key.startswith('__'):
        print(f"{key}: {mat_data[key].shape}")