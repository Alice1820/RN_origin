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
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    
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
    
    
    
