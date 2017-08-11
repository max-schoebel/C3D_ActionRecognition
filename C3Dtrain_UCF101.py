#!/bin/python3
import tensorflow as tf
import C3Dmodel
import InputPipeline as ip
import time
import numpy as np
import os
import threading
import ipdb as pdb
from tensorflow.python.client import timeline
from TimeLiner import TimeLiner
import sys

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
root_ckptdir = "tf_checkpoints"
logdir = "{}/run-{}/".format(root_logdir, now)
ckptdir = "{}/run-{}/".format(root_ckptdir, now)

if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(ckptdir):
    os.makedirs(ckptdir)

TEMPORAL_DEPTH = ip.TEMPORAL_DEPTH
INPUT_WIDTH = ip.INPUT_WIDTH
INPUT_HEIGHT = ip.INPUT_HEIGHT
INPUT_CHANNELS = ip.INPUT_CHANNELS
NUM_CLASSES = ip.NUM_CLASSES

BATCH_SIZE = 40
NUM_GPUS = 2
NUMBER_DATA_THREADS = 7
GPU_QUEUES_CAPACITY = 10
assert(BATCH_SIZE % NUM_GPUS == 0)
EXAMPLES_PER_GPU = int(BATCH_SIZE / NUM_GPUS)

WRITE_TIMELINE = False


def queue_input_placeholders():
    # TODO: Test with variable BATCH_SIZE
    input_placeholder = tf.placeholder(tf.float32,
                                       [EXAMPLES_PER_GPU, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
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
    optimizer = tf.train.AdamOptimizer(1e-04)
    # TODO: capture current adam learning rate
    
    dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training_placeholder')
    tf.add_to_collection("dropout", dropout_placeholder)
    tf.add_to_collection("training", is_training_placeholder)

    tower_gradients = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d'%i), tf.name_scope('Tower%d'%i) as scope:

                with tf.device('/cpu:0'):
                    input_placeholder, output_placeholder, epoch_ended_placeholder = queue_input_placeholders()
                    gpu_queue = tf.FIFOQueue(GPU_QUEUES_CAPACITY, [tf.float32, tf.float32, tf.bool], name='InputQueue{}'.format(i))
                    tf.summary.scalar('queue_fill%d'%i, gpu_queue.size())
                
                    enqueue_op = gpu_queue.enqueue([input_placeholder, output_placeholder, epoch_ended_placeholder])
                    tf.add_to_collection('enqueue', enqueue_op)
                    close_op = gpu_queue.close(cancel_pending_enqueues=True)
                    tf.add_to_collection('close_queue', close_op)

                data, labels, epoch_ended = gpu_queue.dequeue()
                
                network_output = C3Dmodel.inference(data, EXAMPLES_PER_GPU, dropout_placeholder, is_training_placeholder, NUM_CLASSES, collection='network_output')
                xentropy_loss = C3Dmodel.loss(network_output, labels, collection='xentropy_loss')
                
                # train_step = C3Dmodel.train(xentropy_loss, 1e-04, global_step, collection='train_step')
                grads = optimizer.compute_gradients(xentropy_loss)
                tower_gradients.append(grads)
                accuracy_op, accuracy_summary_op = C3Dmodel.accuracy(network_output, labels, collection='accuracy_op')
                
                tf.get_variable_scope().reuse_variables()
    
    grads = average_gradients(tower_gradients)
    train_step = optimizer.apply_gradients(grads, global_step=global_step)
    
    merged_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, my_graph)
    saver = tf.train.Saver()
#    my_graph.finalize()
    
    
def create_train_dict(batch, labels, epoch_ended, gr):
    train_dict = {}
    for tower_index in range(NUM_GPUS):
        start = tower_index * EXAMPLES_PER_GPU
        end = start + EXAMPLES_PER_GPU
        train_dict[gr.get_tensor_by_name('Tower%d/input_placeholder:0'%tower_index)] = batch[start:end]
        train_dict[gr.get_tensor_by_name('Tower%d/output_placeholder:0'%tower_index)] = labels[start:end]
        train_dict[gr.get_tensor_by_name('Tower%d/epoch_ended_placeholder:0'%tower_index)] = epoch_ended
    return train_dict
    
    
def enqueue_gpu_batches(coord, graph, sess, starttime):
    """
    This function is run in a background thread.
        my_graph.get_collection('placeholders')
        Out[3]:
        [<tf.Tensor 'Tower0/input_placeholder:0' shape=(20, 16, 112, 112, 3) dtype=float32>,
         <tf.Tensor 'Tower0/output_placeholder:0' shape=(20, 101) dtype=float32>,
         <tf.Tensor 'Tower0/epoch_ended_placeholder:0' shape=<unknown> dtype=bool>]
    """
    try:
        while not coord.should_stop():
            data, labels, epoch_ended = ip.get_next_batch(BATCH_SIZE)
            feed_dict = create_train_dict(data, labels, epoch_ended, graph)
            sess.run(graph.get_collection('enqueue'), feed_dict=feed_dict)
    except:
        print('Batch abandoned')
        return


def run_training():
    with tf.Session(graph=my_graph, config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # with tf.Session(graph=my_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        assert(tf.get_default_graph() == my_graph)
        sess.run(tf.global_variables_initializer())
        starttime = time.time()

        EPOCHS = 1000
        coord = tf.train.Coordinator()
        enqueue_threads = [threading.Thread(target=enqueue_gpu_batches, args=(coord, my_graph, sess, starttime)) for _ in range(NUMBER_DATA_THREADS)]
        for t in enqueue_threads:
            t.start()
    
        # TIMELINE TEST
        if WRITE_TIMELINE:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            tl = TimeLiner()
            EPOCHS = 1
        else:
            options = None
            run_metadata = None
        ##############
        
        print('------------------------------------------------------------------------------')
        print('Trainable parameters:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
        print('Tensorflow version:', tf.__version__)
        print('------------------------------------------------------------------------------')
        
        feed_dict = {dropout_placeholder : 0.5, is_training_placeholder : False}
    
        for epoch in range(EPOCHS):
            end_epoch = False
            if os.path.isfile('./terminate'):
                break
            while not end_epoch:
                before = time.time()
                _, loss, merged_summ, step, end_epoch = sess.run(
                    [train_step, xentropy_loss, merged_summaries, global_step, epoch_ended],
                    feed_dict=feed_dict,
                    # TIMELINE TEST
                    options=options,
                    run_metadata=run_metadata)
                # pdb.set_trace()
                update_duration = time.time() - before
                before = time.time()
                writer.add_summary(merged_summ, step)
                summary_duration = time.time() - before
                print("Executed global-step  {}  (epoch {})  (time {})".format(step, epoch, time.time()-starttime))
                print(" - Writing summaries took:\t{}".format(summary_duration))
                print(" - Update-step with {} examples took:\t{}".format(BATCH_SIZE, update_duration))
                print(" - Clips/second:\t{}".format(BATCH_SIZE/update_duration))
                print(" - Resulting training loss:\t{}".format(loss))
                
                # TIMELINE TEST
                if WRITE_TIMELINE:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    tl.update_timeline(chrome_trace)
                
            #now = datetime.utcnow().strftime("%Y%m%d%:H%M%S")
            print("------- EPOCH ENDED !!!!! -----------")
            
            # accuracy, accuracy_summary, step = sess.run([accuracy_op, accuracy_summary_op, global_step], feed_dict=test_dict)
            # writer.add_summary(accuracy_summary, step)
            # TODO: Move to tensorflow command FLAGS, i.e. to switch of logging for debug purposes
            saver.save(sess, ckptdir + "/model-{}.ckpt".format(epoch))
            if WRITE_TIMELINE:
                tl.save('./chrome_trace')
            
        coord.request_stop()
        sess.run(my_graph.get_collection('close_queue'))
        coord.join(enqueue_threads)
        
        
        # before = time.time()
        # output = sess.run(network_output, feed_dict=train_dict)
        # print("Forward pass took:{}".format(time.time() - before))

if __name__ == '__main__':
    run_training()
    pass
