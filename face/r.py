import tensorflow as tf
"""
    RNet: Refinement network
    INPUT: 24x24 with 3 channels
    OUTPUT: (1) Facial score (2) coordinates of a bounding box
    [x1, y1, x2, y2]
    [1,0]
    [0,1]
"""
L = tf.keras.layers

class RNet(tf.keras.Model):

    def __init__(self, **kwargs):
        super(RNet, self).__init__(**kwargs)

        """
        Applies 28 convolutional filters of size 3x3 to the 24x24x3 input.
        Stride of 1 means the filter moves one pixel at a time.
        Output size: (24 - 3 + 1) = 22, resulting in a 22x22x28 feature map.
        Parametric ReLU activation, which learns a slope for negative inputs
        Performs max pooling with a 3x3 window and stride of 2, downsampling the feature map.  
        """
        self.conv1 = L.Conv2D(28, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv1")
        self.prelu1 = L.PReLU(shared_axes=[1, 2], name="prelu1")
        self.maxpool1 = L.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same", name="maxpooling1")

        """
        Applies 48 filters of size 3x3 to the 11x11x28 input.
        PReLU2, applies learned activation to the 9x9x48 feature map
        Maxpool results in a 4x4x48 feature
        """
        self.conv2 = L.Conv2D(48, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv2")
        self.prelu2 = L.PReLU(shared_axes=[1, 2], name="prelu2")
        self.maxpool2 = L.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid", name="maxpooling2")

        """PReLU activation to the 3x3x64 feature map."""
        self.conv3 = L.Conv2D(64, kernel_size=(2,2), strides=(1,1), padding="valid", activation="linear", name="conv3")
        self.prelu3 = L.PReLU(shared_axes=[1, 2], name="prelu3")

        """
        [batch, height, width, channels]) to [batch, width, height, channels].
        feature map into a 1D vector: 3 * 3 * 64 = 576 units
        """
        self.permute = L.Permute((2, 1, 3), name="permute")
        self.flatten = L.Flatten(name="flatten3")

        self.fully_connected4 = L.Dense(128, activation="linear", name="fc4")
        self.prelu4 = L.PReLU(name="prelu4")
        self.fully_connected_5_1 = L.Dense(4, activation="linear", name="fc5-1")
        self.fully_connected_5_2 = L.Dense(2, activation="softmax", name="fc5-2")

        self.layers_list = (
        self.conv1, self.prelu1, self.maxpool1, self.conv2, self.prelu2, self.maxpool2, self.conv3, self.prelu3,
        self.permute, self.flatten, self.fully_connected4, self.prelu4, self.fully_connected_5_1, self.fully_connected_5_2)

    def construct(self, size=(None, 24, 24, 3)):
        curr_shape = size
        num_layers = len(self.layers_list)
        i = 0
        while (i < num_layers - 2):
            self.layers_list[i].build(curr_shape)
            curr_shape = self.layers_list[i].compute_output_shape(curr_shape)
            i += 1

        self.fully_connected_5_1.build(curr_shape)
        self.fully_connected_5_1.build(curr_shape)

    def call(self, inputs, *args, **kwargs):
        num_layers = len(self.layers_list)
        i = 0
        while (i < num_layers - 2):
            inputs = self.layers_list[i](inputs)
            i += 1
        regressions = self.fully_connected_5_1(inputs)
        face = self.fully_connected_5_2(inputs)
        return [regressions, face]