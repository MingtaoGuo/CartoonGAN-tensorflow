import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np

epsilon = 1e-8
def conv(name, inputs, nums_out, ksize, strides, padding="SAME", is_D=False):
    # inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    nums_in = int(inputs.shape[-1])
    W = tf.get_variable("W"+name, [ksize, ksize, nums_in, nums_out], initializer=contrib.layers.xavier_initializer(), trainable=True)
    b = tf.get_variable("b"+name, [nums_out], initializer=tf.constant_initializer(0.), trainable=True)
    if is_D:
        return tf.nn.conv2d(inputs, spectral_norm(name, W), [1, strides, strides, 1], padding) + b
    else:
        return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding) + b

def deconv(name, inputs, nums_out, ksize, strides, padding="SAME", is_D=False):
    nums_in = int(inputs.shape[3])
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    W = tf.get_variable("W" + name, [ksize, ksize, nums_in, nums_out], initializer=contrib.layers.xavier_initializer(), trainable=True)
    b = tf.get_variable("b" + name, [nums_out], initializer=tf.constant_initializer(0.), trainable=True)
    inputs = tf.image.resize_nearest_neighbor(inputs, [h*strides, w*strides])
    if is_D:
        return tf.nn.conv2d(inputs, spectral_norm(name, W), [1, 1, 1, 1], padding) + b
    else:
        return tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding) + b

def InstanceNorm(name, inputs):
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        scale = tf.get_variable("scale", shape=mean.shape, initializer=tf.constant_initializer([1.]))
        shift = tf.get_variable("shift", shape=var.shape, initializer=tf.constant_initializer([0.]))
    return (inputs - mean) * scale / tf.sqrt(var + 1e-10) + shift

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(inputs, slope * inputs)

def fully_connected(name, inputs, num_out):
    with tf.variable_scope(name):
        W = tf.get_variable("w"+name, [inputs.shape[-1], num_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b"+name, [num_out], initializer=tf.constant_initializer(0.))
    return tf.matmul(inputs, W) + b

def preprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image - np.array([103.939, 116.779, 123.68])
    else:
        return image - np.array([123.68, 116.779, 103.939])

def spectral_norm(name, w, iteration=1):
    #Spectral normalization which was published on ICLR2018,please refer to "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
    #This function spectral_norm is forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm