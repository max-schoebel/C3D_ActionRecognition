#!/bin/python3
import tensorflow as tf
import C3Dmodel
import time
import numpy as np
import os
import threading
import ipdb as pdb
from tensorflow.python.client import timeline
from TimeLiner import TimeLiner
from DataProvider import UCF101Provider
from DataProvider import CharadesProvider
from EnqueueThread import EnqueueThread
import sys

from datetime import datetime

# root_logdir = "tf_logs"
# root_ckptdir = "tf_checkpoints"
# logdir = "{}/run-{}/".format(root_logdir, now)
# srcdir = logdir + 'src/'
# ckptdir = "{}/run-{}/".format(root_ckptdir, now)

# Save model and training sourcecode to later look up hyperparameters
# if not os.path.exists(srcdir):
#     os.makedirs(srcdir)
# os.system(r'cp ./C3Dtrain.py {}'.format(srcdir))
# os.system(r'cp ./C3Dmodel.py {}'.format(srcdir))
# os.system(r'cp ./DataProvider.py {}'.format(srcdir))
# os.system(r'cp ./videotools.py {}'.format(srcdir))

BATCH_SIZE = 10
NUM_GPUS = 1
NUM_DATA_THREADS = 4
GPU_QUEUES_CAPACITY = 5
assert(BATCH_SIZE % NUM_GPUS == 0)
EXAMPLES_PER_GPU = int(BATCH_SIZE / NUM_GPUS)
LEARNING_RATE = 1e-05

result_filename = './pipeline_performance/ip1gpu.npy'

data_provider = UCF101Provider(BATCH_SIZE, tov_pretraining=False)
TEMPORAL_DEPTH = data_provider.TEMPORAL_DEPTH
INPUT_WIDTH = data_provider.INPUT_WIDTH
INPUT_HEIGHT = data_provider.INPUT_HEIGHT
INPUT_CHANNELS = data_provider.INPUT_CHANNELS
NUM_CLASSES = data_provider.NUM_CLASSES

def queue_input_placeholders():
    # TODO: Test with variable BATCH_SIZE
    input_placeholder = tf.placeholder(
        tf.float32, [EXAMPLES_PER_GPU, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
        name='input_placeholder')
    tf.add_to_collection("placeholders", input_placeholder)
    output_placeholder = tf.placeholder(tf.float32, [EXAMPLES_PER_GPU, NUM_CLASSES], name='output_placeholder')
    tf.add_to_collection("placeholders", output_placeholder)
    epoch_ended_placeholder = tf.placeholder(tf.bool, name='epoch_ended_placeholder')
    tf.add_to_collection("placeholders", epoch_ended_placeholder)
    return input_placeholder, output_placeholder, epoch_ended_placeholder

my_graph = tf.Graph()
with my_graph.as_default():
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    # TODO: capture current adam learning rate
    
    dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training_placeholder')
    tf.add_to_collection("dropout", dropout_placeholder)
    tf.add_to_collection("training", is_training_placeholder)
    
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d'%i), tf.name_scope('Tower%d'%i) as scope:
                
                input_placeholder, output_placeholder, epoch_ended_placeholder = queue_input_placeholders()
                gpu_queue = tf.FIFOQueue(
                    GPU_QUEUES_CAPACITY, [tf.float32, tf.float32, tf.bool], name='InputQueue{}'.format(i))
                tf.summary.scalar('queue_fill%d'%i, gpu_queue.size())
                
                enqueue_op = gpu_queue.enqueue(
                    [input_placeholder, output_placeholder, epoch_ended_placeholder])
                tf.add_to_collection('enqueue', enqueue_op)
                close_op = gpu_queue.close(cancel_pending_enqueues=True)
                tf.add_to_collection('close_queue', close_op)
                
                data, labels, epoch_ended = gpu_queue.dequeue()
                
                with tf.variable_scope('model_replicas'):
                    network_output = C3Dmodel.inference(
                        data, EXAMPLES_PER_GPU, dropout_placeholder, is_training_placeholder, NUM_CLASSES,
                        collection='network_output')
                xentropy_loss, regularization_loss = C3Dmodel.loss(network_output, labels, collection='xentropy_loss', scope=scope)
                
                # train_step = C3Dmodel.train(xentropy_loss, 1e-04, global_step, collection='train_step')
    
    train_step = optimizer.minimize(xentropy_loss, global_step=global_step)

def run_training():
    with tf.Session(graph=my_graph, config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
        # with tf.Session(graph=my_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        assert(tf.get_default_graph() == my_graph)
        sess.run(tf.global_variables_initializer())
        my_graph.finalize()
        
        starttime = time.time()
        
        enqueue_threads = [EnqueueThread(data_provider, my_graph, sess, NUM_GPUS, EXAMPLES_PER_GPU)
                           for _ in range(NUM_DATA_THREADS)]
        for t in enqueue_threads:
            t.start()
        
        options = None
        run_metadata = None
        
        print('------------------------------------------------------------------------------')
        print('Trainable parameters:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
        print('Tensorflow version:', tf.__version__)
        print('------------------------------------------------------------------------------')
        
        feed_dict = {dropout_placeholder : 0.5, is_training_placeholder : False}
        
        end_epoch = False
        print('Filling queues and cooling GPUs')
        for i in range(6):
            print(i*10,'/ 60 s')
            time.sleep(10)
        times = np.zeros(100)
        for i in range(100):
            before = time.time()
            _, loss, step, end_epoch, reg_loss = sess.run(
                [train_step, xentropy_loss, global_step, epoch_ended, regularization_loss],
                feed_dict=feed_dict,
                # TIMELINE TEST
                options=options,
                run_metadata=run_metadata)
            update_duration = time.time() - before
            print("Executed global-step  {}  (time {})".format(step, time.time()-starttime))
            print(" - Update-step with {} examples took:\t{}".format(BATCH_SIZE, update_duration))
            print(" - Clips/second:\t{}".format(BATCH_SIZE/update_duration))
            print(" - Resulting training loss:\t{}  (regularization {})".format(loss, reg_loss))
            times[i] = update_duration
        np.save(result_filename, times)
    
        EnqueueThread.coord.request_stop()
        sess.run(my_graph.get_collection('close_queue'))
        EnqueueThread.coord.join(enqueue_threads)
        
        
        # before = time.time()
        # output = sess.run(network_output, feed_dict=train_dict)
        # print("Forward pass took:{}".format(time.time() - before))

if __name__ == '__main__':
    run_training()
    pass
