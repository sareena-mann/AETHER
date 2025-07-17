from scipy.io import loadmat
mat_data = loadmat('wider_face_train.mat')
print(mat_data.keys())

file_list = mat_data['file_list']
face_bbx_list = mat_data['face_bbx_list']
print("File List Shape:", file_list.shape)
print("First few file names:", file_list[0:3])
print("Bounding Box List Shape:", face_bbx_list.shape)
print("First few bounding boxes:", face_bbx_list[0:3])