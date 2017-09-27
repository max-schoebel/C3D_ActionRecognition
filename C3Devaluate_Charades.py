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
MAX_INPUT_LENGTH = 3
EVAL_INDXS = np.arange(165,100,-5)
OFFSET = 8

input_placeholder = tf.placeholder(
    tf.float32, [None, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
    name='input_placeholder')

# onehot_label_placeholder = tf.placeholder(
#     tf.float32, [NUM_CLASSES]
# )
output = C3Dmodel.inference(input_placeholder, -1, 1, False, NUM_CLASSES, collection='network_output')
# softmax_output = tf.nn.softmax(output)
saver = tf.train.Saver()

# mean_prediction = tf.reduce_mean(output,0)
# correctly_classified = tf.equal(tf.argmax(mean_prediction), tf.argmax(onehot_label_placeholder))

with tf.Session() as sess:
    # path = tf.train.get_checkpoint_state("./tf_checkpoints")
    # graphsaver = tf.train.import_meta_graph(METAGRAPH)

    results_path = CKPT + '/results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    for model_indx in EVAL_INDXS:
        saver.restore(sess, CKPT + '/model-{}.ckpt'.format(model_indx))
        print('Loaded model {}'.format(model_indx))
        test_ended = False
        with open(results_path + '/model-{}-results.txt'.format(model_indx), 'w') as file:
            while not test_ended:
                before = time.time()
                test_video, video_id, test_ended = data_provider.get_next_test_video()
                num_frames = test_video.shape[0]
                starts = np.arange(0, num_frames - TEMPORAL_DEPTH, OFFSET)
                num_clips = len(starts)
                start_tuples = np.array_split(starts, math.ceil(len(starts) / MAX_INPUT_LENGTH))
                outputs = []
                for tuple in start_tuples:
                    clip_list = []
                    for start in tuple:
                        clip_list.append(test_video[start:start+TEMPORAL_DEPTH])
                    input_batch = np.stack(clip_list,0)
                    feed_dict = {input_placeholder : input_batch}
                    out = sess.run(output, feed_dict=feed_dict)
                    outputs.append(out)
                mean_prediction = np.concatenate(outputs, 0)
                mean_prediction = np.mean(mean_prediction, 0)
                file.write(video_id + ' ' + ' '.join(map(str, mean_prediction)) + '\n\n')
                print('Model {} - Video {} - Clips {} - Took {}'.format(model_indx, data_provider.current_test_id-1, num_clips, time.time() - before))
        print('Finished evaluating model {}'.format(model_indx))
        # print(vid, i, ' - ', np.argmax(softmax), np.argmax(label), "    took: " + str(time.time() - before))
        # print(network_output[0].argsort()[-5:][::-1], np.argmax(label))
        # print(np.where(network_output[0].argsort()[::-1] == 0)[0][0], np.argmax(label))
