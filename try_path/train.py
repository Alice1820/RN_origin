#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15:41:24 2017

@author: xifan
"""

import tensorflow as tf
import numpy as np
import model
import os
from six.moves import xrange
import time
from datetime import datetime
from tensorflow.python.platform import gfile
import json
from scipy.misc import imshow, imsave

#import sys
#sys.path.append(['/usr/lib/python2.7.zip', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-linux2', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/lib/python2.7/sie'])

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'train_rev_76', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('num_epoch', 20, """Number of epoches to run.""")
#tf.app.flags.DEFINE_integer('max_steps', 12667, """Number of batches to run per epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 512, """Number of examples per batch""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('if_balance', False, """Whether to use balanced dataset""")
tf.app.flags.DEFINE_boolean('if_shuffle', False, """Whether to shuffle dataset""")
channels = 3

EMBEDDING_DIM = 50
MX_LEN = 64

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3242819
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 699989
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
QUESTION_DICT_LENGTH = 80
ANSWER_DICT_LENGTH = 28

train_json = '../nlvr-master/train/train.json'
train_img_folder = '../nlvr-master/train/images'
test_json = '../nlvr-master/dev/dev.json'
test_img_folder = '../nlvr-master/dev/images'

sentence_log = 'sentence_log.txt'

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

    answer = tf.cast(features['answer'], tf.int32)

    return image, sentence, answer

def generate_batch(batch_size, flag):
#    image, sentence, answer = read_and_decode("/home/zhangxifan/tasks_emb/train_emb.tfrecords")
#    image, sentence, answer = read_and_decode("/home/mi/RelationalReasoning/RelationalReasoning/tasks_emsb/train_emb.tfrecords")
    image, sentence, answer = read_and_decode('/home/zhangxifan/train.tfrecords')
#    image, sentence, answer = read_and_decode('/home/RelationalReasoning/train_balance.tfrecords')

    min_fraction_of_example_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_example_in_queue)
    num_preprocess_threads = 5
    
#    images, sentences_batch, answers_batch = tf.train.shuffle_batch([image, sentence, answer],
#                                                 batch_size = batch_size,
#                                                 num_threads = num_preprocess_threads,
#                                                 capacity = min_queue_examples + 3 * batch_size,
#                                                 min_after_dequeue = 3 * batch_size)
    images, sentences_batch, answers_batch = tf.train.batch([image, sentence, answer],
                                                 batch_size = batch_size,
                                                 num_threads = num_preprocess_threads,
                                                 capacity = min_queue_examples + 3 * batch_size)
#                                                 min_after_dequeue = 3 * batch_size)
#    answers_batch = tf.squeeze(answers_batch, axis = 1)
    return images, sentences_batch, answers_batch

def main(argv = None):
    
#    if gfile.Exists(FLAGS.train_dir):
#        gfile.DeleteRecursively(FLAGS.train_dir)
#        gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable = False)
        # Get images, sentences, labels batch for RN
        images, sentences, answers = generate_batch(batch_size=FLAGS.batch_size, flag='train')
        images = tf.image.resize_bilinear(images, size = [240, 240])
        tf.summary.image('image', tf.expand_dims(images[0], axis = 0))
        print (images)
        print (sentences)
        print (answers)
        
        # Build a graph that computer the logits predictions from inference model
#        logits, cnn_features = model.inference(images, sentences, answers)
        logits, predict_labels = model.inference_demo(images, sentences, answers, FLAGS.batch_size)
        # Calculate loss
        loss = model.loss(logits, answers)
        
        # gradCAM
#        softmax_linear = tf.nn.softmax(logits, dim = -1)[0, :]
##        print (softmax_linear)
##        print (softmax_linear[answers[0]])
##        print (loss)
#        grads = tf.gradients(softmax_linear[answers[0]], cnn_features)
#        print (grads, 'grads')
##        grads_oneimage = tf.expand_dims(grads[0], 0) # summary_image request [batch_size, height, width, nchannels]
#        grads_oneimage = tf.maximum(grads[0][0], 0)
#        grads_max = tf.reduce_max(grads_oneimage)
#        importance_weights = tf.reduce_mean(tf.reduce_mean(grads_oneimage, 0, keep_dims=True), 1, keep_dims=True)
#        print (importance_weights.shape, 'importance_weights.shape')
#        grads_oneimage = tf.multiply(importance_weights, grads_oneimage)
#        grads_oneimage = tf.reduce_mean(grads_oneimage, 2, keep_dims = True) / grads_max * 255
#        print (grads_oneimage, 'grads_oneimage')
#        grads = tf.gradients(softmax_linear[answers[0]], images)
#        backpro_oneimage = tf.maximum(grads[0][0], 0)
#        backpro_oneimage = tf.reduce_mean(backpro_oneimage, 2, keep_dims=True) / tf.reduce_max(backpro_oneimage) * 255
#        grads_oneimage = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(grads_oneimage, 0), size = [320, 480]), axis = 0)
        
#        grad_cam = tf.multiply(grads_oneimage, backpro_oneimage)
#        tf.summary.image('grad-cam', tf.expand_dims(grad_cam, 0))

#        acc = model.accuracy(logits, answers, embedding_matrix)

#        acc = model.accuracy(logits, answers)
        acc = tf.reduce_mean(tf.cast(tf.equal(predict_labels, tf.cast(answers, tf.int64)), tf.float32))
        tf.summary.scalar('accuracy', acc)
        # Build a graph that trains the model with one batch of examples and updates the model parameters
        train_op = model.train(loss, global_step)
        # Create a Saver
        saver = tf.train.Saver(tf.global_variables())
        # Build the summary operation based on the TF collection of Summaries
        summary_op = tf.summary.merge_all()
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth = True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options = gpu_options))
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            print ('Restore from ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        tf.train.start_queue_runners(sess = sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
        start_time = time.time()
        

        
        max_steps = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size)
        for epoch in range(FLAGS.num_epoch):
            average_loss = 0
            for step in xrange(max_steps):
                
#                epoch_num = int(step * FLAGS.batch_size / NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
#                step_tmp = int(step % int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size))
                
                start_time = time.time()
                
    #            s = sess.run([sentences])
    #            print (s)
    #            _, loss_value, accuracy_value = sess.run([train_op, loss, acc])
                answers_value, _, loss_value, accuracy_value, predict_labels_value = sess.run([answers, train_op, loss, acc, predict_labels])
                
    #            print (logits_value[0])
    #            print (softmax_linear_value[0])
                # test dataset
    #            for b in range(FLAGS.batch_size):
    #                imsave('pic/' + str(step) + '_' + str(b) + '.bmp', images_value[b])
    #                for i in range(MX_LEN):
    #    #                    f.write(str(sentences_value[0, i]) + ' ')
    #                    if sentences_value[b, i] != 0 :
    #                        print question_index_to_word[str(sentences_value[b, i])], # python2
    #                print ('\n')
    #                print (answer_index_to_word[str(answers_value[b])])
    ##                print ('\n!')
    
                duration = time.time() - start_time
                
    #            for i in range(FLAGS.batch_size):
    #                # display
    #                prediction_str = str(predict_labels_value[i])
    #                label_str = str(answers_value[i])
    #                print (answer_index_to_word[prediction_str], answer_index_to_word[label_str])
    
                average_loss += loss_value
                
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if step % 1 == 0:
    #                num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = FLAGS.batch_size / duration
    #                sec_per_batch = float(duration)
                    
                    format_str = ('%s: epoch %d, step %d/%d, %.2f, loss = %.5f, average_loss = %.5f, accuracy = %.5f')
                    print (format_str % (datetime.now(), epoch, step, int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size), examples_per_sec, loss_value, average_loss/(step + 1), accuracy_value))
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
                
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = epoch * max_steps + step)
            # eval once
#            print ('*' * 100)
#            eval_acc_avg = 0
#            eval_loss_avg = 0
#            for step in xrange(FLAGS.val_max_steps):
#                answers_value, loss_value, accuracy_value, predict_labels_value = sess.run([answers, loss, acc, predict_labels])
#                eval_acc_avg += accuracy_value
#                eval_loss_avg += loss_value
#                print ('val step: %d/%d, loss: %.5f, accuracy: %.5f' % (step, FLAGS.val_max_steps, loss_value, accuracy_value))
#            print ('%s: epoch %d, loss: %.5f, accuracy: %.5f' % (datetime.now(), epoch, eval_loss_avg, eval_acc_avg))
#            print ('*' * 100)
    
if __name__ == '__main__':
    tf.app.run()
    
