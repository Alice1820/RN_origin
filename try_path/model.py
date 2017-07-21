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

class RN:
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def __call__(self, img, q, is_train):
        g = self.CONV(img, q, is_train = is_train, scope='CONV')
        logits = self.f_phi(g, is_train = is_train, scope='f_phi')
        
        return logits
    
    def generate_path_unit(self, length):
        eight_neighbor = [(0, 1), (1, 0), (1, 1)]
        if length == 1:
            unit_1 = [[(0, 1)], [(1, 0)], [(1, 1)]]
            return unit_1
        else:
            unit = []
            unit_0 = self.generate_path_unit(length = length-1)
            for path in unit_0:
                for x, y in eight_neighbor:
                    x_new = path[-1][0] + x
                    y_new = path[-1][1] + y
                    path_new = []
                    path_new += (path)
                    path_new.append((x_new, y_new))
                    unit.append(path_new)
            
            return unit
            
    def generate_paths(self, height, width, length):
        unit = self.generate_path_unit(length = length)
        PATHS = []
        for x in range(height):
            for y in range(width):
                for path in unit:
    #                print (path)
                    x_end = path[-1][0] + x
                    y_end = path[-1][1] + y
                    if x_end >= 0 and x_end <= height-1 and y_end >= 0 and y_end <= width-1:
                        path_new = []
                        path_new.append((x, y))
                        for x_move, y_move in path:
                            path_new.append((x + x_move, y + y_move))
                        PATHS.append(path_new)
                        path_new.reverse()
                        PATHS.append(path_new)
        return PATHS
    
    def concat_coor(self, o, i, d, batch_size):
        coor = tf.tile(tf.expand_dims(
            [float(int(i / d)) / d, (i % d) / d], axis=0), [batch_size, 1])
        o = tf.concat([o, tf.to_float(coor)], axis=1)
        return o
            
    def g_theta(self, o_1, o_2, o_3, o_4, q, scope='g_theta', reuse=True):
        with tf.variable_scope(scope, reuse=reuse) as scope:
    #        if not reuse: log.warn(scope.name)
            g_1 = fc(tf.concat([o_1, o_2, o_3, o_4, q], axis=1), 256, name='g_1')
            g_2 = fc(g_1, 256, name='g_2')
            g_3 = fc(g_2, 256, name='g_3')
            g_4 = fc(g_3, 256, name='g_4')
            return g_4
    
    # Classifier: takes images as input and outputs class label [B, m]
    def CONV(self, img, q, is_train, scope='CONV'):

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
            PATHS = self.generate_paths(d, d, 3)
            
            all_g = []
    #        PATHS = np.load('paths_7_7_3.npy')
    #        print (PATHS)
            for j in range(len(PATHS)):
                p = PATHS[j]
                o_1 = conv_4[:, p[0][0], p[0][1], :]
                i_1 = p[0][0] * d + p[0][1]
                o_1 = self.concat_coor(o_1, i_1, d, self.batch_size)
                
                o_2 = conv_4[:, p[1][0], p[1][1], :]
                i_2 = p[1][0] * d + p[1][1]
                o_2 = self.concat_coor(o_2, i_2, d, self.batch_size)
                
                o_3 = conv_4[:, p[2][0], p[2][1], :]
                i_3 = p[2][0] * d + p[2][1]
                o_3 = self.concat_coor(o_3, i_3, d, self.batch_size)
                
                o_4 = conv_4[:, p[3][0], p[3][1], :]
                i_4 = p[3][0] * d + p[3][1]
                o_4 = self.concat_coor(o_4, i_4, d, self.batch_size)
                
                if j == 0:
                    g_i_j = self.g_theta(o_1, o_2, o_3, o_4, q, reuse=False)
                else:
                    g_i_j = self.g_theta(o_1, o_2, o_3, o_4, q, reuse=True)
                
                all_g.append(g_i_j)
    
            all_g = tf.stack(all_g, axis=0)
            print (all_g.shape, 'all_g.shape')
            all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
    
            return all_g
    
    def f_phi(self, g, is_train, scope='f_phi'):
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
    # init classes to functions
    lstm = LSTM(layer_size = 128, batch_size = batch_size)
    rn = RN(batch_size = batch_size)
    embedding_function = sentence_embedding(batch_size = batch_size)
    
    embedding, embedding_matrix = embedding_function(questions)
    # reform embedd
    timestep_size = questions.shape[1]
    encoded_sentences = []
    for sentence in tf.unstack(embedding, axis = 1):
        encoded_sentences.append(sentence)
    lstm_encode = lstm(encoded_sentences, timestep_size = timestep_size)[-1]
    
#    g = CONV(PATHS, images, lstm_encode, batch_size = batch_size, is_train = True, scope='CONV')
#    logits = f_phi(g, is_train = True, scope='f_phi')
    
    logits = rn(images, lstm_encode, is_train = True)
    
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
    
    
    
