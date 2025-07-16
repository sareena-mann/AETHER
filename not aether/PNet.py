import tensorflow as tf

#from tensorflow.keras import layers, models

"""
    PNet is the first layer
    INPUT: image of any size
    OUTPUT: (1) Facial score (2) coordinates of a bounding box

"""

L = tf.keras.Layers

"""
keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

"""

class PNet(tf.keras.Model):

    # Instance
    def __init__(self, **kwargs):
        super(PNet, self).__init__(**kwargs)

        #LAYERS
        #Applies 10 different filters to the 3x3 kernel space, kernel moves 1 pixel in each direction, filters are randomly initialized
        self.conv1 = L.Conv2D(10, kernel_size=(3,3), strides=(1,1), padding="valid", activation="linear", name="conv1")
        self.prelu1 = L.PReLU(shared_axes=[1, 2], name="prelu1")
        self.maxpool1 = L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="maxpooling1")



