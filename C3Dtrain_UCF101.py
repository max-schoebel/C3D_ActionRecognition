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
NUM_GPUS = 2

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
    
    optimizer = tf.train.Adam(1e-04)
    tower_gradients = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d'%i), tf.name_scope('Tower%d'%i) as scope:
            
                input_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS])
                output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
                dropout_placeholder = tf.placeholder(tf.float32)
                tf.add_to_collection("placeholders", input_placeholder)
                tf.add_to_collection("placeholders", output_placeholder)
                tf.add_to_collection("placeholders", dropout_placeholder)
                
                network_output = C3Dmodel.inference(input_placeholder, BATCH_SIZE, dropout_placeholder, NUM_CLASSES, collection='network_output')
                xentropy_loss = C3Dmodel.loss(network_output, output_placeholder, collection='xentropy_loss')
                
                # train_step = C3Dmodel.train(xentropy_loss, 1e-04, global_step, collection='train_step')
                grads = optimizer.compute_gradients(xentropy_loss)
                tower_gradients.append(grads)
                accuracy_op, accuracy_summary_op = C3Dmodel.accuracy(network_output, output_placeholder, collection='accuracy_op')
                
                tf.get_variable_scope().reuse_variables()
    
    grads =
    
    merged_summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver()
    
def run_training():
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
            # saver.save(sess, ckptdir + "/model-{}.ckpt".format(epoch))
            
        
        # before = time.time()
        # output = sess.run(network_output, feed_dict=train_dict)
        # print("Forward pass took:{}".format(time.time() - before))

# if __name__ == '__main__':
#     pass
