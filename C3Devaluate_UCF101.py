import tensorflow as tf
import time
import C3Dmodel
from DataProvider import UCF101Provider
import numpy as np
import math
import os

TEMPORAL_DEPTH = UCF101Provider.TEMPORAL_DEPTH
INPUT_WIDTH = UCF101Provider.INPUT_WIDTH
INPUT_HEIGHT = UCF101Provider.INPUT_HEIGHT
INPUT_CHANNELS = UCF101Provider.INPUT_CHANNELS
NUM_CLASSES = UCF101Provider.NUM_CLASSES

CKPT = os.getcwd() + "/tf_checkpoints/run-20170820112437"
MAX_INPUT_LENGTH = 20
EVAL_RANGE = (50, 68)
OFFSET = 4

input_placeholder = tf.placeholder(
    tf.float32, [None, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
    name='input_placeholder')

onehot_label_placeholder = tf.placeholder(
    tf.float32, [NUM_CLASSES]
)
output = C3Dmodel.inference(input_placeholder, -1, 1, False, 101, collection='network_output')
# softmax_output = tf.nn.softmax(output)
saver = tf.train.Saver()

# mean_prediction = tf.reduce_mean(output,0)
# correctly_classified = tf.equal(tf.argmax(mean_prediction), tf.argmax(onehot_label_placeholder))

with tf.Session() as sess:
    # path = tf.train.get_checkpoint_state("./tf_checkpoints")
    # graphsaver = tf.train.import_meta_graph(METAGRAPH)
    data_provider = UCF101Provider(1)
    
    for model_indx in range(EVAL_RANGE[0], EVAL_RANGE[1]+1):
        # saver.restore(sess, CKPT + '/model-{}.ckpt'.format(model_indx))
        saver.restore(sess, './tf_checkpoints/model-12.ckpt')
        test_ended = False
        vid = 0
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
        while not test_ended:
            test_clips, label, test_ended = data_provider.get_next_test_video_clips()
            test_clips = test_clips[::OFFSET]
            test_clips = np.array_split(test_clips, math.ceil(len(test_clips) / MAX_INPUT_LENGTH))
            vid += 1
            before = time.time()
            outputs = []
            for split in test_clips:
                feed_dict = {input_placeholder : split, onehot_label_placeholder : label}
                out = sess.run(output, feed_dict=feed_dict)
                outputs.append(out)
            mean_prediction = np.concatenate(outputs, 0)
            mean_prediction = np.mean(mean_prediction, 0)
            true_label = np.argmax(label)
            predicted_label = np.argmax(mean_prediction)
            confusion_matrix[true_label, predicted_label] += 1
            break
        np.save(CKPT + '/confusion_matrices/model-{}-confmatrix.npy'.format(model_indx), confusion_matrix)
        # print(vid, i, ' - ', np.argmax(softmax), np.argmax(label), "    took: " + str(time.time() - before))
        # print(network_output[0].argsort()[-5:][::-1], np.argmax(label))
        # print(np.where(network_output[0].argsort()[::-1] == 0)[0][0], np.argmax(label))
