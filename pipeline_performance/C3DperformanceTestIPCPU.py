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

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
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

BATCH_SIZE = 40
NUM_GPUS = 1
NUM_DATA_THREADS = 5
GPU_QUEUES_CAPACITY = 20
assert(BATCH_SIZE % NUM_GPUS == 0)
EXAMPLES_PER_GPU = int(BATCH_SIZE / NUM_GPUS)
LEARNING_RATE = 1e-05

result_filename = './pipeline_performance/Perftest_IPCPU.npy'

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
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# with tf.Graph().as_default():
my_graph = tf.Graph()
with my_graph.as_default(), tf.device('/cpu:0'):
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    # TODO: capture current adam learning rate
    
    dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training_placeholder')
    tf.add_to_collection("dropout", dropout_placeholder)
    tf.add_to_collection("training", is_training_placeholder)

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        input_placeholder, output_placeholder, epoch_ended_placeholder = queue_input_placeholders()
        gpu_queue = tf.FIFOQueue(
            GPU_QUEUES_CAPACITY, [tf.float32, tf.float32, tf.bool], name='InputQueue{}'.format(i))
    
        enqueue_op = gpu_queue.enqueue(
            [input_placeholder, output_placeholder, epoch_ended_placeholder])
        tf.add_to_collection('enqueue', enqueue_op)
        close_op = gpu_queue.close(cancel_pending_enqueues=True)
        tf.add_to_collection('close_queue', close_op)

        data, labels, epoch_ended = gpu_queue.dequeue()
        
        network_output = C3Dmodel.inference(
            data, EXAMPLES_PER_GPU, dropout_placeholder, is_training_placeholder, NUM_CLASSES,
            collection='network_output')
        xentropy_loss, regularization_loss = C3Dmodel.loss(network_output, labels, collection='xentropy_loss', scope=scope)
        
        # train_step = C3Dmodel.train(xentropy_loss, 1e-04, global_step, collection='train_step')
    train_step = optimizer.minimize(xentropy_loss)

def run_training():
    with tf.Session(graph=my_graph, config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)) as sess:
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
            # pdb.set_trace()
            update_duration = time.time() - before
            print("Executed global-step  {}  (time {})".format(step, time.time()-starttime))
            print(" - Update-step with {} examples took:\t{}".format(BATCH_SIZE, update_duration))
            print(" - Clips/second:\t{}".format(BATCH_SIZE/update_duration))
            print(" - Resulting training loss:\t{}  (regularization {})".format(loss, reg_loss))
            times[i] = update_duration
                
        print("------- EPOCH ENDED !!!!! -----------")
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
