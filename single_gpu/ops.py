import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(input, output_shape, is_train, activation_fn=tf.nn.relu,
           k_h=3, k_w=3, s_h=2, s_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s_h, s_w, 1], padding='SAME')
#        biases = tf.get_variable('biases', [output_shape],
#                                 initializer=tf.constant_initializer(0.0))
#        activation = activation_fn(conv + biases)
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                          decay=0.9, is_training=is_train,
                                          updates_collections=None)
        activation = activation_fn(bn)
    return activation


def fc(input, output_shape, activation_fn=tf.nn.relu, name="fc"):
    output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
    return output
