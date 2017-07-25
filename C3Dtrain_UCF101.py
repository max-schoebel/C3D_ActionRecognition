import tensorflow as tf
import C3Dmodel
import InputPipeline as ip
import time
import numpy as np
import os

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

BATCH_SIZE = 1

def model_variable(name, shape, wd):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(0,0.1))
    return var

# with tf.Graph().as_default():
my_graph = tf.Graph()
with my_graph.as_default():
    # When training on multiple GPUs, variable scope is needed to maintain a single copy of these variables
    weights = {
        'wconv1': model_variable('wconv1', [3, 3, 3, 3, 64], 0.0005),
        'wconv2': model_variable('wconv2', [3, 3, 3, 64, 128], 0.0005),
        'wconv3a': model_variable('wconv3a', [3, 3, 3, 128, 256], 0.0005),
        'wconv3b': model_variable('wconv3b', [3, 3, 3, 256, 256], 0.0005),
        'wconv4a': model_variable('wconv4a', [3, 3, 3, 256, 512], 0.0005),
        'wconv4b': model_variable('wconv4b', [3, 3, 3, 512, 512], 0.0005),
        'wconv5a': model_variable('wconv5a', [3, 3, 3, 512, 512], 0.0005),
        'wconv5b': model_variable('wconv5b', [3, 3, 3, 512, 512], 0.0005),
        'wfully1': model_variable('wfully1', [8192, 4096], 0.0005),
        'wfully2': model_variable('wfully2', [4096, 4096], 0.0005),
        'wout': model_variable('wout', [4096, NUM_CLASSES], 0.0005)
    }

    biases = {
        'bconv1': model_variable('bconv1', [64], 0.000),
        'bconv2': model_variable('bconv2', [128], 0.000),
        'bconv3a': model_variable('bconv3a', [256], 0.000),
        'bconv3b': model_variable('bconv3b', [256], 0.000),
        'bconv4a': model_variable('bconv4a', [512], 0.000),
        'bconv4b': model_variable('bconv4b', [512], 0.000),
        'bconv5a': model_variable('bconv5a', [512], 0.000),
        'bconv5b': model_variable('bconv5b', [512], 0.000),
        'bfully1': model_variable('bfully1', [4096], 0.000),
        'bfully2': model_variable('bfully2', [4096], 0.000),
        'bout': model_variable('bout', [NUM_CLASSES], 0.000),
    }

    input_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS])
    output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
    dropout_placeholder = tf.placeholder(tf.float32)
    tf.add_to_collection("placeholders", input_placeholder)
    tf.add_to_collection("placeholders", output_placeholder)
    tf.add_to_collection("placeholders", dropout_placeholder)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    network_output = C3Dmodel.inference(input_placeholder, BATCH_SIZE, weights, biases, dropout_placeholder, collection='network_output')
    xentropy_loss = C3Dmodel.loss(network_output, output_placeholder, collection='xentropy_loss')
    train_step = C3Dmodel.train(xentropy_loss, 1e-04, global_step, collection='train_step')
    accuracy_op, accuracy_summary_op = C3Dmodel.accuracy(network_output, output_placeholder, collection='accuracy_op')
    
    merged_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver()
    

# with tf.Session(graph=my_graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session(graph=my_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    EPOCHS = 1000
    # testvideos = np.load('testset.npy')
    # testlabels = np.load('testlabels.npy')
    # test_dict = {
    #     input_placeholder: testvideos,
    #     output_placeholder: testlabels,
    #     dropout_placeholder: 1.0
    # }
    
    print('------------------------------------------------------------------------------')
    print('Trainable parameters:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
    print('Tensorflow version:', tf.__version__)
    print('------------------------------------------------------------------------------')
    
    for epoch in range(EPOCHS):
        epoch_ended = False
        while not epoch_ended:
            before = time.time()
            batch, labels, epoch_ended = ip.get_next_batch(BATCH_SIZE)
            duration = time.time() - before
            print("Getting examples took {}s".format(duration))
            train_dict = {
                input_placeholder: batch,
                output_placeholder: labels,
                dropout_placeholder: 0.5  # == keep probability
            }
            before = time.time()
            _, loss, merged_summ, step = sess.run([train_step, xentropy_loss, merged_summaries, global_step], feed_dict=train_dict)
            duration = time.time() - before
            writer.add_summary(merged_summ, step)
            print("   Current step is:    {}".format(step))
            print("   Single update-step with {} examples took: {}".format(BATCH_SIZE, duration))
            print("   Resulting training loss: {}".format(loss))
            
            epoch_ended = True
            
        print("------- EPOCH ENDED!!!!! -----------")
        # accuracy, accuracy_summary, step = sess.run([accuracy_op, accuracy_summary_op, global_step], feed_dict=test_dict)
        # writer.add_summary(accuracy_summary, step)
        saver.save(sess, ckptdir + "/model-{}.ckpt".format(epoch))
        
    
    # before = time.time()
    # output = sess.run(network_output, feed_dict=train_dict)
    # print("Forward pass took:{}".format(time.time() - before))

# if __name__ == '__main__':
#     pass
