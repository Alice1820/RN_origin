#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:49:47 2017

@author: xifan
"""
import json
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
#from unidecode import unidecode

train_file = 'CLEVR_v1.0/questions/CLEVR_train_questions.json'
train_image_path = 'CLEVR_v1.0/images/train'
val_file = 'CLEVR_v1.0/questions/CLEVR_val_questions.json'
val_image_path = 'CLEVR_v1.0/images/val'

my_answer_word_to_index = {}
my_answer_index_to_word = {}
my_question_word_to_index = {}
my_question_index_to_word = {}

MX_LEN = 64

def create_dict(file, images_path):
    print ("Loading data from %s" % (file))
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    js_answer_word = 0
    js_question_word = 0
    f = open(file, 'r')
    for l in f:
        jn = json.loads(l)
        q_list = jn['questions']
        js = 0
        for q in q_list:
            js += 1
            with tf.Graph().as_default():
                print ('%d / %d' % (js, len(q_list))) 
    #            image_index = q['image_index']
    #            question_index = q['question_index']
                image_filename = q['image_filename']
                answer = q['answer']
                question = q['question']
                
                image = imread(images_path + '/' + image_filename, mode = 'RGB')
                
#                imsave('pic/' + str(js) + '.bmp', image)
                print (question)
                print (answer)
                # Build the dict
                if answer not in my_answer_word_to_index:
                    my_answer_word_to_index[answer] = js_answer_word
                    my_answer_index_to_word[js_answer_word] = answer
                    js_answer_word += 1
                
                answer_index = my_answer_word_to_index[answer]
                
                question_index = np.zeros([MX_LEN], dtype = 'int64')
                question = str(question) # python2
                
                sequence = tf.contrib.keras.preprocessing.text.text_to_word_sequence(question)
#                print (tf.contrib.keras.preprocessing.text.text_to_word_sequence(question))
                for i in range(len(sequence)):
                    if sequence[i] not in my_question_word_to_index:
                        js_question_word += 1
                        my_question_word_to_index[sequence[i]] = js_question_word
                        my_question_index_to_word[js_question_word] = sequence[i]
                    question_index[i] = my_question_word_to_index[sequence[i]]
                
#                print (question)
#                print (question_index)
#                print (answer)
#                print (answer_index)
#                print (int(question_index[i]) for i in range(MX_LEN))
                index = [i for i in range(MX_LEN)]
#                print (index)
#                print (question_index[index])
                example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
#                "cnn_features": tf.train.Feature(bytes_list=tf.train.BytesList(value=[cnn_features.tobytes()])),
                "question": tf.train.Feature(bytes_list=tf.train.BytesList(value=[question_index.tobytes()])),
                "answer": tf.train.Feature(int64_list=tf.train.Int64List(value=[answer_index])),
                }))
    
                writer.write(example.SerializeToString())
    writer.close()

create_dict(train_file, train_image_path)
#load_val_data(val_file, val_image_path)

# Save dicts to json
word_to_index_obj = json.dumps(my_answer_word_to_index)
index_to_word_obj = json.dumps(my_answer_index_to_word)
FileObj = open('my_answer_word_to_index.json', 'w')
FileObj.write(word_to_index_obj)
FileObj.close()
FileObj = open('my_answer_index_to_word.json', 'w')
FileObj.write(index_to_word_obj)
FileObj.close()
word_to_index_obj = json.dumps(my_question_word_to_index)
index_to_word_obj = json.dumps(my_question_index_to_word)
FileObj = open('my_question_word_to_index.json', 'w')
FileObj.write(word_to_index_obj)
FileObj.close()
FileObj = open('my_question_index_to_word.json', 'w')
FileObj.write(index_to_word_obj)
FileObj.close()

