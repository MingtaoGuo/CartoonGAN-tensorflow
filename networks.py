from ops import *
import scipy.io as sio

class generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, reuse=False):
        #just use the network of CycleGAN's generator
        with tf.variable_scope(self.name, reuse=reuse):
            inputs = tf.nn.relu(InstanceNorm("IN1", conv("c7s1-32", inputs, 32, 7, 1)))
            inputs = tf.nn.relu(InstanceNorm("IN2", conv("d64", inputs, 64, 3, 2)))
            inputs = tf.nn.relu(InstanceNorm("IN3", conv("d128", inputs, 128, 3, 2)))
            for i in range(6):
                temp = inputs
                inputs = tf.nn.relu(InstanceNorm("INB"+str(i), conv("R_conv1"+str(i), inputs, 128, 3, 1)))
                inputs = conv("R_conv2"+str(i), inputs, 128, 3, 1)
                inputs = temp + inputs
            inputs = tf.nn.relu(InstanceNorm("IN5", deconv("u64", inputs, 64, 3, 2)))
            inputs = tf.nn.relu(InstanceNorm("IN6", deconv("u32", inputs, 32, 3, 2)))
            inputs = tf.nn.tanh(conv("c7s1-3", inputs, 3, 7, 1))
            return (inputs + 1.) * 127.5

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            inputs = leaky_relu(conv("conv1", inputs, 32, 3, 1, "SAME", True))
            inputs = leaky_relu(conv("conv2", inputs, 64, 3, 2, "SAME", True))
            inputs = leaky_relu(InstanceNorm("IN1", conv("conv3", inputs, 128, 3, 1, "SAME", True)))
            inputs = leaky_relu(conv("conv4", inputs, 128, 3, 2, "SAME", True))
            inputs = leaky_relu(InstanceNorm("IN2", conv("conv5", inputs, 256, 3, 1, "SAME", True)))
            inputs = leaky_relu(InstanceNorm("IN3", conv("conv6", inputs, 256, 3, 1, "SAME", True)))
            inputs = conv("conv7", inputs, 1, 3, 1, "SAME", True)
            inputs = tf.layers.flatten(inputs)
            inputs = fully_connected("logits", inputs, 1)
            return tf.nn.sigmoid(inputs)

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class VGG:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        inputs = preprocess(tf.reverse(inputs, [-1]))
        vgg_para = sio.loadmat("./vgg_para//vgg.mat")
        layers = vgg_para["layers"]
        with tf.variable_scope(self.name):
            for i in range(37):
                if layers[0, i][0, 0]["type"] == "conv":
                    w = layers[0, i][0, 0]["weights"][0, 0]
                    b = layers[0, i][0, 0]["weights"][0, 1]
                    with tf.variable_scope(str(i)):
                        inputs = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b
                        if layers[0, i][0, 0]["name"][0] == "conv4_4":
                            return inputs
                elif layers[0, i][0, 0]["type"] == "relu":
                    inputs = tf.nn.relu(inputs)
                else:
                    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")


