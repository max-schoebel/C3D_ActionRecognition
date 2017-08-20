import tensorflow as tf
import time
import C3Dmodel
from DataProvider import UCF101Provider
import numpy as np

TEMPORAL_DEPTH = UCF101Provider.TEMPORAL_DEPTH
INPUT_WIDTH = UCF101Provider.INPUT_WIDTH
INPUT_HEIGHT = UCF101Provider.INPUT_HEIGHT
INPUT_CHANNELS = UCF101Provider.INPUT_CHANNELS
NUM_CLASSES = UCF101Provider.NUM_CLASSES

CKPT = "./tf_checkpoints/model-74.ckpt"

input_placeholder = tf.placeholder(
    tf.float32, [None, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
    name='input_placeholder')
onehot_label_placeholder = tf.placeholder(
    tf.float32, [NUM_CLASSES]
)
# output_placeholder = tf.placeholder(tf.float32, [1, NUM_CLASSES], name='output_placeholder')
output = C3Dmodel.inference(input_placeholder, -1, 1, False, 101, collection='network_output')
softmax_output = tf.nn.softmax(output)
saver = tf.train.Saver()

mean_prediction = tf.reduce_mean(output,0)
correctly_classified = tf.equal(tf.argmax(mean_prediction), tf.argmax(onehot_label_placeholder))

with tf.Session() as sess:
    # path = tf.train.get_checkpoint_state("./tf_checkpoints")
    # graphsaver = tf.train.import_meta_graph(METAGRAPH)
    saver.restore(sess, CKPT)
    
    data_provider = UCF101Provider(1)
    
    test_ended = False
    vid = 0
    classification_results = []
    while not test_ended:
        test_clips, label, test_ended = data_provider.get_next_test_video_clips()
        vid += 1
        before = time.time()
        feed_dict = {input_placeholder : test_clips[::8], onehot_label_placeholder : label}
        # network_output, softmax, corr_class = sess.run([output, softmax_output, correctly_classified], feed_dict=feed_dict)
        corr_class = sess.run(correctly_classified, feed_dict=feed_dict)
        # print(network_output)
        # print(softmax)
        print(vid, corr_class, 'No Test-Clips: {}'.format(len(test_clips[::8])))
        classification_results.append(int(corr_class))
        # print(vid, i, ' - ', np.argmax(softmax), np.argmax(label), "    took: " + str(time.time() - before))
        # print(network_output[0].argsort()[-5:][::-1], np.argmax(label))
        # print(np.where(network_output[0].argsort()[::-1] == 0)[0][0], np.argmax(label))
    
    
    # input_placeholder = graph.get_operation_by_name("Placeholder")
    # out_placeholder = graph.get_operation_by_name("Placeholder_1")
    # dropout_placeholder = graph.get_operation_by_name("Placeholder_2")
    #
    # global_step = tf.train.get_global_step()
    #
    # test_dict = {
    #             input_placeholder: testvideos,
    #             out_placeholder: testlabels,
    #             dropout_placeholder: 1  # == keep probability
    #             }
    #
    # print(sess.run(global_step))
