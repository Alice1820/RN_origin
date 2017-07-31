#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:05:26 2017

@author: xifan
"""

import tensorflow as tf
import numpy as np
import model_29
import os
from six.moves import xrange
import time
from datetime import datetime
from tensorflow.python.platform import gfile
import json
from scipy.misc import imshow, imsave
import re
import random
#import sys
#sys.path.append(['/usr/lib/python2.7.zip', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-linux2', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/lib/python2.7/site-packages'])

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'train_multi_fixbn', """Directory where to write event logs and checkpoint.""")
#tf.app.flags.DEFINE_string('train_dir', 'train', """Directory where to load model.""")
tf.app.flags.DEFINE_integer('num_epoch', 100, """Number of epoches to run.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'train_multi_fixbn', """Directory where to load model""")
#tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of examples per batch""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('if_balance', False, """Whether to use balanced dataset""")
tf.app.flags.DEFINE_boolean('if_shuffle', True, """Whether to shuffle dataset""")

channels = 3

EMBEDDING_DIM = 50
MX_LEN = 64
MX_LEN_CUT = 50

QUESTION_DICT_LENGTH = 80
ANSWER_DICT_LENGTH = 28

IMAGE_ROWS = 240
IMAGE_COLOMNS = 240

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350
LEARNING_RATE_DECAY_FACTOR = 0.5
INITIAL_LEARNING_RATE = 2.5e-5

train_json = '../nlvr-master/train/train.json'
train_img_folder = '../nlvr-master/train/images'
test_json = '../nlvr-master/dev/dev.json'
test_img_folder = '../nlvr-master/dev/images'

sentence_log = 'sentence_log.txt'
TOWER_NAME = 'tower'

num_gpus = 2

# Load dicts
f = open('my_question_index_to_word.json', 'r')
question_index_to_word = json.load(f)
f.close()
f = open('my_question_word_to_index.json', 'r')
question_word_to_index = json.load(f)
f.close()
QUESTION_DICT_LENGTH = len(question_word_to_index)
f = open('my_answer_index_to_word.json', 'r')
answer_index_to_word = json.load(f)
f.close()
f = open('my_answer_word_to_index.json', 'r')
answer_word_to_index = json.load(f)
f.close()
ANSWER_DICT_LENGTH = len(answer_word_to_index)


def read_and_decode(filename):
    # Build a filename_queue according to filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'question': tf.FixedLenFeature([], tf.string),
                                           'answer': tf.FixedLenFeature([], tf.int64),
                                       })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [320, 480, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    
    sentence = tf.decode_raw(features['question'], tf.int64)
#    sentence = tf.reshape(features['question'], [MX_LEN])
    sentence = tf.reshape(sentence, [MX_LEN])
    sentence = sentence[:50]
    sentence = tf.cast(sentence, tf.int32)

    answer = tf.add(tf.cast(features['answer'], tf.int32), 1)

    return image, sentence, answer

def generate_batch(batch_size, flag):
    if FLAGS.if_balance:
        image, sentence, answer = read_and_decode('/home/zhangxifan/train_balance.tfrecords')
    else:
        image, sentence, answer = read_and_decode('/home/zhangxifan/train.tfrecords')
#    image, sentence, answer = read_and_decode('/home/RelationalReasoning/train_balance.tfrecords')

#    min_fraction_of_example_in_queue = 0.4
#    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_example_in_queue)
    min_queue_examples = 10000
    num_preprocess_threads = 5
    
    if FLAGS.if_shuffle:
        images, sentences_batch, answers_batch = tf.train.shuffle_batch([image, sentence, answer],
                                                 batch_size = batch_size,
                                                 num_threads = num_preprocess_threads,
                                                 capacity = min_queue_examples + 3 * batch_size,
                                                 min_after_dequeue = min_queue_examples)
    else:
        images, sentences_batch, answers_batch = tf.train.batch([image, sentence, answer],
                                                     batch_size = batch_size,
                                                     num_threads = num_preprocess_threads,
                                                     capacity = min_queue_examples + 3 * batch_size)
    return images, sentences_batch, answers_batch

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    Args:
        batch_size: The batch size will be baked into _hat placeholders.
    Returns:
        images_placeholder.
        y_i_placeholder.
        x_hat_placeholder.
        y_hat_placeholder.
    """

    images_pl = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_ROWS, IMAGE_COLOMNS, 3], name='images')
    sentences_pl = tf.placeholder(tf.int32, shape=[batch_size, MX_LEN_CUT], name='sentences')
    answers_pl = tf.placeholder(tf.float32, shape=[batch_size], name='answers')

    return images_pl, sentences_pl, answers_pl

def tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
#    images, sentences, answers = placeholder_inputs(FLAGS.batch_size)
    images, sentences, answers = generate_batch(batch_size=FLAGS.batch_size, flag='train')
    # preprocess
    images = tf.image.resize_bilinear(images, size = [128, 128])
    images = tf.image.resize_image_with_crop_or_pad(images, target_height = 136, target_width = 136)
    images = tf.random_crop(images, size = [FLAGS.batch_size, 128, 128, 3])
    rota_range = 0.05 # rads
    rota_range = rota_range * (random.random() - 0.5)
    images = tf.contrib.image.rotate(images, rota_range)
    
    tf.summary.image('image', tf.expand_dims(images[0], axis = 0))
#    tower_inputs = (images, sentences, answers)
    print (images)
    print (sentences)
    print (answers)
    
    logits, predict_labels = model_29.inference_demo(images, sentences, answers, FLAGS.batch_size)
    
    # Calculate the total loss for the current tower
    acc = tf.reduce_mean(tf.cast(tf.equal(predict_labels, tf.cast(answers, tf.int64)), tf.float32))
    acc_name = re.sub('%s_[0-9]*/' % 'tower', '', acc.op.name)
    tf.summary.scalar(acc_name + '(raw)', acc)

#    total_loss = model.loss(logits, answers)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = model_29.loss(logits, answers)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    

    return total_loss, acc

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train_multi_gpu():
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    """
    if FLAGS.if_balance:
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3242819

    else:
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 699989
        
    with tf.Graph().as_default(), tf.device('/cpu:0'):
#        decay_steps = 20000
#        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        global_step = tf.Variable(0, trainable = False)
        
#        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                        global_step,
#                                        decay_steps,
#                                        LEARNING_RATE_DECAY_FACTOR,
#                                        staircase = True)
        
    #    loss_average_op = _add_loss_summaries(total_loss)
        
        # Compute gradients
        opt = tf.train.AdamOptimizer(learning_rate = INITIAL_LEARNING_RATE)
        # Calculate the gradients for each model tower.
        tower_grads = []
#        tower_inputs = []
#        tower_losses = []
#        tower_acc = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(num_gpus):
              with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                  # Calculate the loss for one tower of the CIFAR model. This function
                  # constructs the entire CIFAR model but shares the variables across
                  # all towers.
                  loss, acc = tower_loss(scope)
    #                  tower_losses.append(loss)
    #                  tower_acc.append(acc)
    #                  tower_inputs.append(tower_input)
                  # Reuse variables for the next tower.
    #              if i != 0:
                  tf.get_variable_scope().reuse_variables()
                  # Retain the summaries from the final tower.
                  summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                  # Calculate the gradients for the batch of data on this CIFAR tower.
                  grads = opt.compute_gradients(loss)
                  # Keep track of the gradients across all towers.
                  tower_grads.append(grads)

#        loss_avg = tf.reduce_mean(tf.stack(tower_losses))
#        acc_avg = tf.reduce_mean(tf.stack(tower_acc))
        
        with tf.device('/gpu:0'):
    #        metric_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #        metric_averages_op = metric_averages.apply([loss_avg, acc_avg])
            loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', loss.op.name)
            acc_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', acc.op.name)
    
    #        summaries.append(tf.summary.scalar(loss_name, metric_averages.average(loss)))
    #        summaries.append(tf.summary.scalar(acc_name, metric_averages.average(acc)))
            summaries.append(tf.summary.scalar(loss_name, loss))
            summaries.append(tf.summary.scalar(acc_name, acc))
    
    #        print (tower_grads)
            grads = average_gradients(tower_grads)
            print (grads)
            print ('*' * 80)
#            summaries.append(tf.summary.scalar('learning_rate', lr))
    
            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                    
            # Apply gradients
            apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
    
            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            
    #        for grad, var in grads:
    #            if grad is not None:
    #                tf.summary.histogram(var.op.name + '/gradients', grad)
            
            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
    #        for item in tf.trainable_variables():
    #            print (item.name)
            
            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)
        
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge(summaries)
        
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth = True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options = gpu_options))

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            print ('Restore from' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    
        max_steps = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/(FLAGS.batch_size * num_gpus))
        for epoch in range(FLAGS.num_epoch):
            average_loss = 0
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            for step in xrange(max_steps):
                
#                epoch_num = int(step * FLAGS.batch_size / NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
#                step_tmp = int(step % int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size))
#                #test
                start_time_0 = time.time()
                _, loss_value, accuracy_value = sess.run([train_op, loss, acc])
                duration = time.time() - start_time_0
#                print (duration, 'loss_time')
##                start_time = time.time()
##                _ = sess.run(grads)
##                duration = time.time() - start_time
##                print (duration, 'grads_time')
#                start_time = time.time()
#                _ = sess.run(train_op)
#                duration = time.time() - start_time
#                print (duration, 'train_time')
#                duration = time.time() - start_time_0
#                print (duration, 'time')
##                start_time = time.time()
#                
#    #            s = sess.run([sentences])
#    #            print (s)
#    #            _, loss_value, accuracy_value = sess.run([train_op, loss, acc])
#                _, loss_value, accuracy_value = sess.run([train_op, loss, acc])
#                
#    #            print (logits_value[0])
#    #            print (softmax_linear_value[0])
#                # test dataset
#    #            for b in range(FLAGS.batch_size):
#    #                imsave('pic/' + str(step) + '_' + str(b) + '.bmp', images_value[b])
#    #                for i in range(MX_LEN):
#    #    #                    f.write(str(sentences_value[0, i]) + ' ')
#    #                    if sentences_value[b, i] != 0 :
#    #                        print question_index_to_word[str(sentences_value[b, i])], # python2
#    #                print ('\n')
#    #                print (answer_index_to_word[str(answers_value[b])])
#    ##                print ('\n!')
#    
#                duration = time.time() - start_time
                
    #            for i in range(FLAGS.batch_size):
    #                # display
    #                prediction_str = str(predict_labels_value[i])
    #                label_str = str(answers_value[i])
    #                print (answer_index_to_word[prediction_str], answer_index_to_word[label_str])
    
                average_loss += loss_value
                
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if step % 1 == 0:
    #                num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = FLAGS.batch_size * num_gpus / duration
    #                sec_per_batch = float(duration)
                    
                    format_str = ('%s: epoch %d, step %d/%d, %.2f, loss = %.5f, average_loss = %.5f, accuracy = %.5f')
                    print (format_str % (datetime.now(), epoch, step, max_steps, examples_per_sec, loss_value, average_loss/(step + 1), accuracy_value))
    #                format_str = ('%s: step %d, loss = %d')
    #                print (format_str % (datetime.now(), step, loss_value))
                
                if step % 100 == 0:
                    # print sentences to file
    #                f = open(sentence_log, 'a')
    #                f.write('step: ' + str(step) + ' ')
    #                for i in range(MX_LEN):
    ##                    f.write(str(sentences_value[0, i]) + ' ')
    #                    if sentences_value[0, i] != 0 :
    #                        f.write(question_index_to_word[str(sentences_value[0, i])] + ' ')
    #                f.write(answer_index_to_word[str(answers_value[0])])
    #                imsave('pic/' + str(step) + '.bmp', images_value[0])
    #                f.write('\n')
    #                f.close()
                    
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                
                if step % 2000 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = epoch * max_steps + step)
        coord.request_stop()
        coord.join(threads)

def main(argv = None):
    
#    if gfile.Exists(FLAGS.train_dir):
#        gfile.DeleteRecursively(FLAGS.train_dir)
#        gfile.MakeDirs(FLAGS.train_dir)
    train_multi_gpu()
    
    
if __name__ == '__main__':
    tf.app.run()
    
