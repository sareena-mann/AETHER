import tensorflow as tf
from tensorflow.python.eager.context import set_log_device_placement

"""
    PNet is the first layer
    INPUT: image of any size
    OUTPUT: (1) Facial score (2) coordinates of a bounding box
    [x1, y1, x2, y2]
    [1,0]
    [0,1]

    Conv1 + PReLU1 + MaxPool1: Extracts low-level features and downsamples.
    Conv2 + PReLU2: Extracts mid-level features.
    Conv3 + PReLU3: Extracts high-level features.
    Conv4_1: Outputs bounding box regressions (4 values per location).
    Conv4_2: Outputs classification scores (2 values per location).
"""


"""
conv1: Input (12, 12, 3) → Output (10, 10, 10) (since kernel_size=(3,3), strides=(1,1), padding="valid" reduces spatial dimensions by 2).
prelu1: No shape change, output (10, 10, 10).
maxpool1: Input (10, 10, 10) → Output (5, 5, 10) (since pool_size=(2,2), strides=(2,2), padding="same" halves the spatial dimensions).
conv2: Input (5, 5, 10) → Output (3, 3, 16) (since kernel_size=(3,3), padding="valid" reduces spatial dimensions by 2).
prelu2: No shape change, output (3, 3, 16).
conv3: Input (3, 3, 16) → Output (1, 1, 32) (since kernel_size=(3,3), padding="valid" reduces spatial dimensions by 2).
prelu3: No shape change, output (1, 1, 32).
conv4_1: Input (1, 1, 32) → Output (1, 1, 4) (for bounding box regression).
conv4_2: Input (1, 1, 32) → Output (1, 1, 2) (for classification scores).



"""

L = tf.keras.layers

class PNet(tf.keras.Model):

    # Instance
    def __init__(self, **kwargs):
        super(PNet, self).__init__(**kwargs)

        #LAYERS
        #Applies 10 different filters to the 3x3 kernel space, kernel moves 1 pixel in each direction, filters are randomly initialized
        #output size = input size - kernel size + 1
        # The conv1 layer in PNet extracts 10 low-level feature maps using 3x3 kernels, focusing on edges, textures, and simple patterns in the input image. Low-level visual info
        self.conv1 = L.Conv2D(10, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv1")
        self.prelu1 = L.PReLU(shared_axes=[1, 2], name="prelu1")

        # Downsamples the feature map, reducing spatial dimensions while retaining important features, making the network
        # more robust to translations and reducing computation.
        self.maxpool1 = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="maxpooling1")
        self.conv2 = L.Conv2D(16, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv2")
        self.prelu2 = L.PReLU(shared_axes=[1, 2], name="prelu2")
        self.conv3 = L.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv3")
        self.prelu3 = L.PReLU(shared_axes=[1, 2], name="prelu3")
        self.conv4_1 = L.Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="linear", name="conv4-1")
        self.conv4_2 = L.Conv2D(2, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="linear")

        self.layers_list = (self.conv1, self.prelu1, self.maxpool1, self.conv2, self.prelu2, self.conv3, self.prelu3, self.conv4_1, self.conv4_2)


    def construct(self, size=(None, 12, 12, 3)):
        curr_shape = size
        num_layers = len(self.layers_list)
        i = 0
        while (i < num_layers - 2):
            self.layers_list[i].build(curr_shape)
            curr_shape = self.layers_list[i].compute_output_shape(curr_shape)

            i += 1
        self.conv4_1.build(curr_shape)
        self.conv4_2.build(curr_shape)



    def call(self, inputs):
        num_layers = len(self.layers_list)
        i = 0
        while (i < num_layers - 2):
            inputs = self.layers_list[i](inputs)
            i += 1
        regressions = self.conv4_1(inputs)
        face = self.conv4_2(inputs)
        return [regressions, face]