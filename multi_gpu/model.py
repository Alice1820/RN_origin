#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:28:50 2017

@author: xifan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.slim as slim
import numpy as np
import re
from tensorflow.python.ops.nn_ops import max_pool
from scipy.misc import imresize
import tensorflow.contrib.slim as slim
from ops import conv2d, fc

#from ops import conv2d, fc

#from train import GLOVE
TOWER_NAME = 'tower'
weight_file = 'vgg16_weights.npz'

NUM_CLASSES = 50
#batch_size = 128
MX_LEN = 64 # the max length of a sentence
EMBEDDING_DIM = 50 # word look-up embeddings dim

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 700000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350
LEARNING_RATE_DECAY_FACTOR = 0.5
INITIAL_LEARNING_RATE = 2.5e-4

QUESTION_DICT_LENGTH = 80
ANSWER_DICT_LENGTH = 28

conv_info = np.array([64, 64, 64, 64])

def _activation_summary(x):
    """
    Helper to create summaries for activation
    Creates a summary that provides a histogram of activation
    Creates a summary that measure the sparsity of activaiton
    """
    
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(tensor_name + '/activation', x)


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer = initializer)
        
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    """
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev = stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var

#def concat_coor(o, i, d):
#    coor = tf.tile(tf.expand_dims(
#        [float(int(i / d)) / d, (i % d) / d], axis=0), [BATCH_SIZE, 1])
#    o = tf.concat([o, tf.to_float(coor)], axis=1)
#    return o
#        
#def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
#    with tf.variable_scope(scope, reuse=reuse) as scope:
##        if not reuse: log.warn(scope.name)
#        g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
#        g_2 = fc(g_1, 256, name='g_2')
#        g_3 = fc(g_2, 256, name='g_3')
#        g_4 = fc(g_3, 256, name='g_4')
#        return g_4
#
## Classifier: takes images as input and outputs class label [B, m]
#def CONV(img, q, scope='CONV', is_train):
#    with tf.variable_scope(scope) as scope:
##        log.warn(scope.name)
#        conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
#        conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
#        conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
#        conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')
#
#        # eq.1 in the paper
#        # g_theta = (o_i, o_j, q)
#        # conv_4 [B, d, d, k]
#        d = conv_4.get_shape().as_list()[1]
#        all_g = []
#        for i in range(d*d):
#            o_i = conv_4[:, int(i / d), int(i % d), :]
#            o_i = concat_coor(o_i, i, d)
#            for j in range(d*d):
#                o_j = conv_4[:, int(j / d), int(j % d), :]
#                o_j = concat_coor(o_j, j, d)
#                if i == 0 and j == 0:
#                    g_i_j = g_theta(o_i, o_j, q, reuse=False)
#                else:
#                    g_i_j = g_theta(o_i, o_j, q, reuse=True)
#                all_g.append(g_i_j)
#
#        all_g = tf.stack(all_g, axis=0)
#        all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
#        return all_g
#
#def f_phi(g, scope='f_phi', is_train):
#    with tf.variable_scope(scope) as scope:
##        log.warn(scope.name)
#        fc_1 = fc(g, 256, name='fc_1')
#        fc_2 = fc(fc_1, 256, name='fc_2')
#        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
#        fc_3 = fc(fc_2, ANSWER_DICT_LENGTH, activation_fn=None, name='fc_3')
#        return fc_3
def concat_coor(o, i, d, batch_size):
    coor = tf.tile(tf.expand_dims(
        [float(int(i / d)) / d, (i % d) / d], axis=0), [batch_size, 1])
    o = tf.concat([o, tf.to_float(coor)], axis=1)
    return o
        
def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
    with tf.variable_scope(scope, reuse=reuse) as scope:
#        if not reuse: log.warn(scope.name)
        g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
        g_2 = fc(g_1, 256, name='g_2')
        g_3 = fc(g_2, 256, name='g_3')
        g_4 = fc(g_3, 256, name='g_4')
        return g_4

# Classifier: takes images as input and outputs class label [B, m]
def CONV(img, q, batch_size, is_train, scope='CONV'):
    with tf.variable_scope(scope) as scope:
#        log.warn(scope.name)
        conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
        conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
        conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
        conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')

        # eq.1 in the paper
        # g_theta = (o_i, o_j, q)
        # conv_4 [B, d, d, k]
        print (conv_4.shape, 'conv_4.shape')
        d = conv_4.get_shape().as_list()[1]
        all_g = []
        for i in range(d*d):
            o_i = conv_4[:, int(i / d), int(i % d), :]
            o_i = concat_coor(o_i, i, d, batch_size)
            for j in range(d*d):
                o_j = conv_4[:, int(j / d), int(j % d), :]
                o_j = concat_coor(o_j, j, d, batch_size)
                if i == 0 and j == 0:
                    g_i_j = g_theta(o_i, o_j, q, reuse=False)
                else:
                    g_i_j = g_theta(o_i, o_j, q, reuse=True)
                all_g.append(g_i_j)

        all_g = tf.stack(all_g, axis=0)
        all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
        return all_g

def f_phi(g, is_train, scope='f_phi'):
    with tf.variable_scope(scope) as scope:
#        log.warn(scope.name)
        fc_1 = fc(g, 256, name='fc_1')
        fc_2 = fc(fc_1, 256, name='fc_2')
        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
        fc_3 = fc(fc_2, ANSWER_DICT_LENGTH, activation_fn=None, name='fc_3')
        return fc_3
    
class cnn_layers:
    def __init__(self, batch_size, layer_sizes, num_channels=1):
        """
        Builds a CNN to produce embeddings
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        
        self.layers = {}
        
        assert len(self.layer_sizes)==4, "layer_sizes should be a list of length 4"

    def __call__(self, image_input, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 320, 480, 1]
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """
        
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        with tf.variable_scope('g', reuse=self.reuse):

            with tf.variable_scope('conv_layers'):
                with tf.variable_scope('g_conv1'):
                    self.g_conv1_encoder = tf.layers.conv2d(image_input, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                       padding='VALID')
                    self.g_conv1_encoder = leaky_relu(self.g_conv1_encoder, name='outputs')
                    self.g_conv1_encoder = tf.contrib.layers.batch_norm(self.g_conv1_encoder, updates_collections=None, decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    self.g_conv1_encoder = max_pool(self.g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    self.g_conv1_encoder = tf.nn.dropout(self.g_conv1_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv2'):
                    self.g_conv2_encoder = tf.layers.conv2d(self.g_conv1_encoder, self.layer_sizes[1], [3, 3], strides=(1, 1),
                                                       padding='VALID')
                    self.g_conv2_encoder = leaky_relu(self.g_conv2_encoder, name='outputs')
                    self.g_conv2_encoder = tf.contrib.layers.batch_norm(self.g_conv2_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    self.g_conv2_encoder = max_pool(self.g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    self.g_conv2_encoder = tf.nn.dropout(self.g_conv2_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv3'):
                    self.g_conv3_encoder = tf.layers.conv2d(self.g_conv2_encoder, self.layer_sizes[2], [3, 3], strides=(2, 2),
                                                       padding='VALID')
                    self.g_conv3_encoder = leaky_relu(self.g_conv3_encoder, name='outputs')
                    self.g_conv3_encoder = tf.contrib.layers.batch_norm(self.g_conv3_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    self.g_conv3_encoder = max_pool(self.g_conv3_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    self.g_conv3_encoder = tf.nn.dropout(self.g_conv3_encoder, keep_prob=keep_prob)

                with tf.variable_scope('g_conv4'):
                    self.g_conv4_encoder = tf.layers.conv2d(self.g_conv3_encoder, self.layer_sizes[3], [3, 3], strides=(2, 2),
                                                       padding='VALID')
                    self.g_conv4_encoder = leaky_relu(self.g_conv4_encoder, name='outputs')
                    self.g_conv4_encoder = tf.contrib.layers.batch_norm(self.g_conv4_encoder, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    self.g_conv4_encoder = max_pool(self.g_conv4_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    self.g_conv4_encoder = tf.nn.dropout(self.g_conv4_encoder, keep_prob=keep_prob)
            
                conv4_oneimage = tf.expand_dims(tf.maximum(self.g_conv4_encoder[0,], 0), 0) # summary_image request [batch_size, height, width, nchannels]
                conv4_max = tf.reduce_max(conv4_oneimage)
                conv4_oneimage = tf.reduce_mean(conv4_oneimage, 3, keep_dims = True) / conv4_max * 255
                tf.summary.image('conv4%d' % 0, conv4_oneimage)
                image_oneimage = tf.expand_dims(image_input[0], 0)
                tf.summary.image('image%d' % 0, image_oneimage)
                
            self.g_conv_encoder = self.g_conv4_encoder
#            g_conv_encoder = tf.contrib.layers.flatten(g_conv_encoder)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        
        self.layers['g_conv1'] = self.g_conv1_encoder
        self.layers['g_conv2'] = self.g_conv2_encoder
        self.layers['g_conv3'] = self.g_conv3_encoder
        self.layers['g_conv4'] = self.g_conv4_encoder
        
        return self.g_conv_encoder

class MLP_g:
    """
    Build a MLP
    :params batch_size
    :params object_set: MLP inputs, [batch_size, 128 + 128]
    :returns: outputs MLP outputs, [batch_size, 256]
    """
    def __init__(self, hidden_size, batch_size):
        self.reuse = False
        self.hidden_size = hidden_size
        self.batch_size = batch_size
    
    def __call__(self, batch_size, object_set):
        with tf.variable_scope('MLP_g'):
            # output = tf.get_variable([batch_size, n_class], tf.constant_initializer(0.0))
            outputs = []
            for b in range(batch_size):
                # fc 1
                if b == 0:
                    reuse = self.reuse
                else:
                    reuse = True
                with tf.variable_scope('fc1', reuse = reuse) as scope:
                    fc1 = tf.contrib.layers.fully_connected(object_set, 256)
                    fc1 = tf.contrib.layers.batch_norm(fc1, updates_collections=None,
                                                               decay=0.99,
                                                               scale=True, center=True)
                    fc1 = tf.nn.relu(fc1)
                    
                # fc 2
                with tf.variable_scope('fc2', reuse = reuse) as scope:
                    fc2 = tf.contrib.layers.fully_connected(fc1, 256)
                    fc2 = tf.contrib.layers.batch_norm(fc2, updates_collections=None,
                                                               decay=0.99,
                                                               scale=True, center=True)
                    fc2 = tf.nn.relu(fc2)
                    
                # fc 3
                with tf.variable_scope('fc3', reuse = reuse) as scope:
                    fc3 = tf.contrib.layers.fully_connected(fc2, 256)
                    fc3 = tf.contrib.layers.batch_norm(fc3, updates_collections=None,
                                                               decay=0.99,
                                                               scale=True, center=True)
                    fc3 = tf.nn.relu(fc3)
                    
                # fc 4
                with tf.variable_scope('fc4', reuse = reuse) as scope:
                    fc4 = tf.contrib.layers.fully_connected(fc3, 256)
                    fc4 = tf.contrib.layers.batch_norm(fc4, updates_collections=None,
                                                                   decay=0.99,
                                                                   scale=True, center=True)
                    fc4 = tf.nn.relu(fc4)
#                    
                output = tf.reduce_mean(fc4, 0)
                print (output.shape, 'output.shape')
                outputs.append(output)
            outputs = tf.stack(outputs)
            print (outputs.shape, 'outputs.shape')
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MLP_g')
        return outputs

class MLP_f:
    """
    Build a MLP
    :params batch_size
    :params object_set: MLP inputs, [batch_size, (7*7)^2, 512*2 + 512*2]
    :returns: outputs MLP outputs, [batch_size, 256]
    """
    def __init__(self, hidden_size, batch_size):
        self.reuse = False
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = {}
    
    def __call__(self, batch_size, object_set):
        with tf.variable_scope('MLP_f'):
            # output = tf.get_variable([batch_size, n_class], tf.constant_initializer(0.0))
            outputs = []
            with tf.variable_scope('fc1', reuse = self.reuse) as scope:
                self.fc1 = tf.contrib.layers.fully_connected(object_set, 256)
                self.fc1= tf.contrib.layers.batch_norm(self.fc1, updates_collections=None,
                                                           decay=0.99,
                                                           scale=True, center=True)
                self.fc1 = tf.nn.relu(self.fc1)
                
            # fc 2
            with tf.variable_scope('fc2', reuse = self.reuse) as scope:
                self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 256)
                self.fc2= tf.contrib.layers.batch_norm(self.fc2, updates_collections=None,
                                                           decay=0.99,
                                                           scale=True, center=True)
                self.fc2 = tf.nn.relu(self.fc2)
                self.fc2 = tf.nn.dropout(self.fc2, keep_prob = 0.5)
            # fc 3
            with tf.variable_scope('fc3', reuse = self.reuse) as scope:
                self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 28)
                self.fc3= tf.contrib.layers.batch_norm(self.fc3, updates_collections=None,
                                                           decay=0.99,
                                                           scale=True, center=True)
                self.fc3 = tf.nn.relu(self.fc3)
#                fc3 = tf.contrib.layers.batch_norm(fc3, updates_collections=None,
#                                                               decay=0.99,
#                                                               scale=True, center=True)
#                self.fc3 = tf.nn.softmax(self.fc3)
            
            outputs = tf.stack(self.fc3)
            print (outputs.shape, 'outputs.shape')
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MLP_f')
        
        self.layers['fc1'] = self.fc1
        self.layers['fc2'] = self.fc2
        self.layers['fc3'] = self.fc3
        
        return outputs

class BidirectionalLSTM:
    def __init__(self, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes

    def __call__(self, inputs, name, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [timestep_size, batch_size, length]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs
        """
#        print (inputs.shape, 'lstm inputs.shape')
        with tf.name_scope('bid-lstm' + name), tf.variable_scope('bid-lstm', reuse=self.reuse):
            fw_lstm_cells = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                             for i in range(len(self.layer_sizes))]
            bw_lstm_cells = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                             for i in range(len(self.layer_sizes))]

            outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                fw_lstm_cells,
                bw_lstm_cells,
                inputs,
                dtype=tf.float32
            )
#        print (outputs.shape, 'lstm outputs.shape')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bid-lstm')
        return outputs


class LSTM:
    """"""
    def __init__(self, layer_size, batch_size):
        self.hidden_size = layer_size
        self.batch_size = batch_size
        self.reuse = False
    
    def __call__(self, images, timestep_size):
        """
        Args:
            images: unstacked and stack again, [MX_LEN, batch_size, EMBEDDING_DIM]
        Returns:
            outputs: Returns the LSTM outputs, as well as the forward and backward hidden states, 
                    [MX_LEN, batch_size, hidden_size]
        """
        self.timestep_size = timestep_size
        lstm_inputs = images
        with tf.variable_scope('lstm'):
            outputs = []
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias = 1.0)
            state = cell.zero_state(self.batch_size, dtype = tf.float32)
            for timestep in range(self.timestep_size):
                if timestep == 0:
                    reuse = self.reuse
                else:   reuse = True
#                print (timestep, 'timestep')
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias = 1.0, reuse = reuse)
                input_tmp = lstm_inputs[timestep]
                (cell_output, state) = cell(input_tmp, state) # [batch_size, EMBEDDING_DIM]
                outputs.append(cell_output)
        outputs = tf.stack(outputs)
        print (outputs.shape, 'lstm_outputs.shape')
        self.reuse = True
        return outputs


def generate_MLP_inputs(inputs, gen_encode):
    """for each two objects in object set, create a pair
    Args:
        inputs: cnn_features, returned from cnn_layers, [batch_size, 5, 3, 64]
        gen_code: [batch_size, 64]
    """
    batch_size = BATCH_SIZE
    height = int(inputs.shape[1])
    width = int(inputs.shape[2])
    nchannels = int(inputs.shape[3])
    object_set = []
    """
    for b in range(batch_size):
        print (b)
        object_set_minibatch = []
        for i in range(height * width):
            for j in range(height * width):
                if i == j:
                    pass
                i_1 = int(i/width)
                j_1 = int(i % width)
                i_2 = int(j/width)
                j_2 = int(j % width)
#                i_1 = i // width
#                j_1 = i - i_1 * width
#                i_2 = j // width
#                j_2 = j - i_2 * width
                # object_set[b, i * height * width + j, : ] = tf.concat(0, [inputs[b, i_1, j_1, : ], inputs[b, i_2, j_2, : ]])
                #object_set[b, i * height * width + j, 0 : 64 ] = inputs[b, i_1, j_1, : ]
                #object_set[b, i * height * width + j, 64 : ] = inputs[b, i_2, j_2, : ]
                tmp = tf.concat([inputs[b, i_1, j_1, : ], inputs[b, i_2, j_2, : ]], axis = 0)
#                print (tmp.shape)
                object_set_minibatch.append(tf.concat([tmp, gen_encode[b]], axis = -1))
#                print ("(%d, %d)" % (i, j))
        object_set_minibatch = tf.stack(object_set_minibatch)
#        print (object_set_minibatch.shape, 'object_set_minibatch.shape')
        object_set.append(object_set_minibatch)
    object_set = tf.stack(object_set)
    print (object_set.shape, 'object_set.shape')
    return object_set"""
    
#    inputs1 = tf.expand_dims(inputs, axis = 1)
    inputs1 = tf.stack([inputs for _ in range(height * width)], axis = 1)
#    inputs2 = tf.expand_dims(inputs, axis = 3)
    inputs2 = tf.stack([inputs for _ in range(height * width)], axis = 3)
#    gen_encode = tf.squeeze(gen_encode)
#    gen_encode = tf.expand_dims(gen_encode, axis = 1)
#    print (gen_encode.shape, 'gen_encode.shape')
    gen_encode = tf.stack([gen_encode for _ in range(height * width * height * width)], axis = 1)
    print (gen_encode.shape, 'gen_encode.shape')
    print (inputs1.shape, 'inputs1.shape')
    print (inputs2.shape, 'inputs2.shape')
    inputs1 = tf.reshape(inputs1, [-1, height * width * height * width, nchannels])
    inputs2 = tf.reshape(inputs2, [-1, height * width * height * width, nchannels])
    object_set = tf.concat([inputs1, inputs2, gen_encode], axis = -1)
    print (gen_encode.shape, 'gen_encode.shape')
    print (object_set.shape, 'object_set.shape')
    return object_set

#def generate_MLP_inputs_2(inputs, gen_encode):
#    """only adjencent objects
#    """
#    batch_size = BATCH_SIZE
#    height = 7
#    width = 7
#
#    object_set = []
#    for b in range(batch_size):
#        object_set_minibatch = []
#        for i in range(height):
#            for j in range(width):
##                i_1 = i // width
##                j_1 = i - i_1 * width
##                i_2 = j // width
##                j_2 = j - i_2 * width
#                # object_set[b, i * height * width + j, : ] = tf.concat(0, [inputs[b, i_1, j_1, : ], inputs[b, i_2, j_2, : ]])
#                #object_set[b, i * height * width + j, 0 : 64 ] = inputs[b, i_1, j_1, : ]
#                #object_set[b, i * height * width + j, 64 : ] = inputs[b, i_2, j_2, : ]
#                if i != 0:
#                tmp = tf.concat([inputs[b, i_1, j_1, : ], inputs[b, i_2, j_2, : ]], axis = 0)
##                print (tmp.shape)
#                object_set_minibatch.append(tf.concat([tmp, gen_encode], axis = -1))
#                print ("(%d, %d)" % (i, j))
#        object_set_minibatch = tf.stack(object_set_minibatch)
#        print (object_set_minibatch.shape, 'object_set_minibatch.shape')
#        object_set.append(object_set_minibatch)
#    object_set = tf.stack(object_set)
#    print (object_set.shape, 'object_set.shape')
#    return object_set

class sentence_embedding:
    def __init__(self, batch_size):
        self.reuse = False
        self.batch_size = batch_size
        self.dict_length = QUESTION_DICT_LENGTH
    
    def __call__(self, sentences):
        """Embed the sentences, answers with EMBEDDING_DIM
        Args:
            sentences:[batch_size, MX_LEN]
        Returns:
            embedding: [batch_size, MX_LEN, EMBEDDING_DIM]
        """
        with tf.variable_scope('embedding', reuse = self.reuse):
            clevr_on_GLOVE = np.load('clevr_question_on_GLOVE.npy')
            init = tf.constant_initializer(clevr_on_GLOVE)
            embedding_matrix = tf.get_variable('embedding_matrix', shape = [self.dict_length + 1, EMBEDDING_DIM], dtype = tf.float32, 
                                               initializer = init)
            print (sentences.shape, 'sentences.shape')
            onehot_sentences = tf.one_hot(sentences, self.dict_length + 1) # [batch_size, MX_LEN, dict_length]
            print (onehot_sentences.shape, 'onehot_sentences.shape')
            # Embed sentences with embedding_matrix
            embedding = []
            for b in range(self.batch_size):
                embedding_tmp = tf.matmul(onehot_sentences[b], embedding_matrix)
                embedding.append(embedding_tmp)
            embedding = tf.stack(embedding) # [batch_size, MX_LEN, EMBEDDING_DIM]
        self.reuse = True
        
        return embedding, embedding_matrix

def inference_demo(images, questions, answers, batch_size):
    
    lstm = LSTM(layer_size = 128, batch_size = batch_size)
    embedding_function = sentence_embedding(batch_size = batch_size)
    
    embedding, embedding_matrix = embedding_function(questions)
    timestep_size = questions.shape[1]
    encoded_sentences = []
    for sentence in tf.unstack(embedding, axis = 1):
        encoded_sentences.append(sentence)
    lstm_encode = lstm(encoded_sentences, timestep_size = timestep_size)[-1]
    
    g = CONV(images, lstm_encode, batch_size = batch_size, is_train = True, scope='CONV')
    logits = f_phi(g, is_train = True, scope='f_phi')
    all_preds = tf.nn.softmax(logits)
    print (all_preds.shape, 'all_preds.shape')
    predict_labels = tf.argmax(all_preds, axis = 1)
    print (predict_labels.shape, 'predict_labels.shape')
    
    return logits, predict_labels

    
def inference(images, ws_batch, answers):
    """
    Build the RN model
    Args:
        images: Images returned from generate_batch, [batch_size, 320, 480, 3]
        ws_batch: Sentences, [batch_size, MX_LEN = 50]
        answers: [batch_size], 0~ANSWER_DICT_LENGTH-1
    Returns:
        logits
    """
    batch_size = batch_size
#    vgg = vgg16(images)
    mlp_g = MLP_g(hidden_size = 256, batch_size = batch_size)
    mlp_f = MLP_f(hidden_size = 256, batch_size = batch_size)
    
    # 128 units LSTM for question processing
#    lstm = BidirectionalLSTM(layer_sizes = [64], batch_size = batch_size)
    lstm = LSTM(layer_size = 128, batch_size = batch_size)
    
    # initiate 4-layer CNN
    cnn = cnn_layers(batch_size, num_channels = 3 , layer_sizes=[64, 64, 64 ,64])
    embedding_function = sentence_embedding(batch_size = batch_size)
    
    # trainable CNN
    cnn_features = cnn(images, training = True)
    print (cnn_features.shape, 'cnn_features.shape')
    
    # Embed questions
    embedding, embedding_matrix = embedding_function(ws_batch)
    encoded_sentences = []
    for sentence in tf.unstack(embedding, axis = 1):
        encoded_sentences.append(sentence)
    print (embedding.shape, 'embedding.shape') # [batch_size, MX_LEN, EMBEDDING_DIM]
    print (len(encoded_sentences), 'len(encoded_sentences)') # MX_LEN
    print (encoded_sentences[0].shape, 'encoded_sentences[0].shape') # [batch_size, EMBEDDING_DIM]
    # LSTM
#    timestep_size = tf.argmin(ws_batch, axis = -1)
    timestep_size = ws_batch.shape[1]
    lstm_encode = lstm(encoded_sentences, timestep_size = timestep_size)[-1]
    print (lstm_encode, 'lstm_encode.shape')

    object_set = generate_MLP_inputs(cnn_features, lstm_encode)
    print (object_set.shape, 'object_set.shape')
    
    # MLP g & f
    encode_g = mlp_g(batch_size, object_set)
    encode_g = tf.reduce_mean(encode_g, axis = 1)
    print (encode_g.shape, 'encode_g.shape')
    encode_f = mlp_f(batch_size, encode_g)
    print (encode_f.shape, 'encode_f.shape')
    
    # softmax
    with tf.variable_scope('linear_layer'):
        weights = _variable_with_weight_decay('weights', shape = [ANSWER_DICT_LENGTH, ANSWER_DICT_LENGTH], stddev = 1/ANSWER_DICT_LENGTH, wd = 0.0)
        biases = _variable_on_cpu('biases', [ANSWER_DICT_LENGTH], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(encode_f, weights), biases)
        _activation_summary(softmax_linear)
#    predict_labels = accuracy(logits = encode_f, answers = answers)
#    cam = gradCAM(cnn_features = cnn_features[0], mlp_f = encode_f[0], img = images[0], layer_name = 'g_conv4', predicted_class = predict_labels[0], nb_classes = ANSWER_DICT_LENGTH)
    
    return softmax_linear, cnn_features

def accuracy(logits, answers):
    """Calculate accuracy
    Args:
        logits: [batch_size, ANSWER_DICT_LENGTH]
        answers: correct labels [batch_size], each with a number, indicating the index of word in dict
    Returns:
        accuracy
    """
    # Get the predicted labels, [batch_size]
    with tf.variable_scope('accuracy'):
        predict_labels = tf.argmax(tf.nn.softmax(logits, dim = -1), axis = 1)
#        tf.summary.text('predictions', predict_labels)
#        tf.summary.text('answers', answers)
#        tf.summary.text('softmax', tf.nn.softmax(logits, dim = -1))
    
    correct_predictions = tf.equal(predict_labels, tf.cast(answers, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    tf.summary.scalar('accuracy', accuracy)
    
    return accuracy


def loss(logits, answers):
    """Add L2loss to all trainable variables
    """
    # Embed answers with embedding_matrix
    onehot_answers = tf.one_hot(answers, ANSWER_DICT_LENGTH) # [batch_size, answer_dict_length]
    
    print (logits.shape, 'logits.shape')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = onehot_answers, name = 'cross_entropy_per_example')
#    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = answers, logits = logits, name = 'cross_entropy_per_example')
#    mse = tf.div(tf.reduce_mean(tf.square(tf.subtract(logits, embedding))), 2, name ="mse")
#    tf.add_to_collection('losses', mse)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name +' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))
    
    return loss_averages_op

def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    """
    decay_steps = 20000
#    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase = True)
    tf.summary.scalar('learning_rate', lr)
    
    loss_average_op = _add_loss_summaries(total_loss)
    
    # Compute gradients
    with tf.control_dependencies([loss_average_op]):
        #opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(learning_rate = lr)
    grads = opt.compute_gradients(total_loss)
    # Apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    for item in tf.trainable_variables():
        print (item.name)
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
    return train_op
    
    
    
