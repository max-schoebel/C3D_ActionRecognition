import tensorflow as tf
import time
import C3Dmodel
from DataProvider import UCF101Provider, CharadesProvider
import numpy as np
import math
import os

data_provider = CharadesProvider()
TEMPORAL_DEPTH = data_provider.TEMPORAL_DEPTH
INPUT_WIDTH = data_provider.INPUT_WIDTH
INPUT_HEIGHT = data_provider.INPUT_HEIGHT
INPUT_CHANNELS = data_provider.INPUT_CHANNELS
NUM_CLASSES = data_provider.NUM_CLASSES

CKPT = os.getcwd() + "/tf_checkpoints/run-20170828082324"
MAX_INPUT_LENGTH = 20
EVAL_INDXS = np.arange(0,165,5)
OFFSET = 8

input_placeholder = tf.placeholder(
    tf.float32, [None, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
    name='input_placeholder')

onehot_label_placeholder = tf.placeholder(
    tf.float32, [NUM_CLASSES]
)
output = C3Dmodel.inference(input_placeholder, -1, 1, False, NUM_CLASSES, collection='network_output')
# softmax_output = tf.nn.softmax(output)
saver = tf.train.Saver()

# mean_prediction = tf.reduce_mean(output,0)
# correctly_classified = tf.equal(tf.argmax(mean_prediction), tf.argmax(onehot_label_placeholder))

with tf.Session() as sess:
    # path = tf.train.get_checkpoint_state("./tf_checkpoints")
    # graphsaver = tf.train.import_meta_graph(METAGRAPH)
    
    for model_indx in EVAL_INDXS:
        saver.restore(sess, CKPT + '/model-{}.ckpt'.format(model_indx))
        print('Loaded model {}'.format(model_indx))
        test_ended = False
        vid = 0
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
        results = np.zeros((len(data_provider.training_vidpath_label_tuples[data_provider.current_split]), data_provider.NUM_CLASSES))
        while not test_ended:
            before = time.time()
            test_clips, label, test_ended = data_provider.get_next_test_video_clips()
            test_clips = test_clips[::OFFSET]
            num_clips = len(test_clips)
            test_clips = np.array_split(test_clips, math.ceil(len(test_clips) / MAX_INPUT_LENGTH))
            outputs = []
            for split in test_clips:
                feed_dict = {input_placeholder : split, onehot_label_placeholder : label}
                out = sess.run(output, feed_dict=feed_dict)
                outputs.append(out)
            mean_prediction = np.concatenate(outputs, 0)
            mean_prediction = np.mean(mean_prediction, 0)
            results[vid] = mean_prediction
            true_label = np.argmax(label)
            predicted_label = np.argmax(mean_prediction)
            confusion_matrix[true_label, predicted_label] += 1
            print('Model {} - Video {} - Frames {} - Took {}'.format(model_indx, vid+1, num_clips, time.time() - before))
            vid += 1
        confmatrix_path = CKPT + '/confusion_matrices/model-{}-confmatrix.npy'.format(model_indx)
        results_path = CKPT + 'results/model-{}-results.npy'.format(model_indx)
        np.save(confmatrix_path, confusion_matrix)
        np.save(results_path, results)
        print('Finished evaluating model {}'.format(model_indx))
        print('Confusion matrix saved to {}'.format(confmatrix_path))
        # print(vid, i, ' - ', np.argmax(softmax), np.argmax(label), "    took: " + str(time.time() - before))
        # print(network_output[0].argsort()[-5:][::-1], np.argmax(label))
        # print(np.where(network_output[0].argsort()[::-1] == 0)[0][0], np.argmax(label))
