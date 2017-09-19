import threading
import tensorflow as tf
import sys

class EnqueueThread(threading.Thread):
    """
    This class defines a thread to run in background and fill the GPU-queues.
        my_graph.get_collection('placeholders')
        Out[3]:
        [<tf.Tensor 'Tower0/input_placeholder:0' shape=(20, 16, 112, 112, 3) dtype=float32>,
         <tf.Tensor 'Tower0/output_placeholder:0' shape=(20, 101) dtype=float32>,
         <tf.Tensor 'Tower0/epoch_ended_placeholder:0' shape=<unknown> dtype=bool>]
    """
    lock = threading.Lock()
    coord = tf.train.Coordinator()
    
    def __init__(self, data_provider, graph, session, num_gpus, examples_per_gpu):
        threading.Thread.__init__(self)
        self.data_provider = data_provider
        self.graph = graph
        self.sess = session
        self.num_gpus = num_gpus
        self.examples_per_gpu = examples_per_gpu
    
    def run(self):
        try:
            while not self.coord.should_stop():
                data, labels, epoch_ended = self.data_provider.get_next_training_batch(self.lock)
                feed_dict = {}
                
                for tower_index in range(self.num_gpus):
                    start = tower_index * self.examples_per_gpu
                    end = start + self.examples_per_gpu
                    
                    # TODO: make independent of tensor naming
                    data_tensor = self.graph.get_tensor_by_name('Tower%d/input_placeholder:0'%tower_index)
                    feed_dict[data_tensor] = data[start:end]
                    
                    label_tensor = self.graph.get_tensor_by_name('Tower%d/output_placeholder:0'%tower_index)
                    feed_dict[label_tensor] = labels[start:end]
                    
                    epoch_ended_tensor = self.graph.get_tensor_by_name('Tower%d/epoch_ended_placeholder:0'%tower_index)
                    feed_dict[epoch_ended_tensor] = epoch_ended
                
                self.sess.run(self.graph.get_collection('enqueue'), feed_dict=feed_dict)
        except:
            print('Batch abandoned')
            print(sys.exc_info())


if __name__ == '__main__':
    ### TEST CODE ###
    from DataProvider import UCF101Provider
    from DataProvider import CharadesProvider
    from DataProvider import KineticsPretrainProvider
    import numpy as np
    from videotools import are_equal
    import time
    NUM_GPUS = 1
    BATCH_SIZE = 8
    EXAMPLES_PER_GPU = 4
    GPU_QUEUES_CAPACITY = 16
    NUM_DATA_THREADS = 2

    data_provider = KineticsPretrainProvider(BATCH_SIZE)
    TEMPORAL_DEPTH = data_provider.TEMPORAL_DEPTH
    INPUT_WIDTH = data_provider.INPUT_WIDTH
    INPUT_HEIGHT = data_provider.INPUT_HEIGHT
    INPUT_CHANNELS = data_provider.INPUT_CHANNELS
    NUM_CLASSES = data_provider.NUM_CLASSES

    def queue_input_placeholders():
        # TODO: Test with variable BATCH_SIZE
        input_placeholder = tf.placeholder(
            tf.float32, [EXAMPLES_PER_GPU, 3, TEMPORAL_DEPTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
            name='input_placeholder')
        tf.add_to_collection("placeholders", input_placeholder)
        output_placeholder = tf.placeholder(tf.float32, [EXAMPLES_PER_GPU, NUM_CLASSES], name='output_placeholder')
        tf.add_to_collection("placeholders", output_placeholder)
        epoch_ended_placeholder = tf.placeholder(tf.bool, name='epoch_ended_placeholder')
        tf.add_to_collection("placeholders", epoch_ended_placeholder)
        return input_placeholder, output_placeholder, epoch_ended_placeholder
    
    
    my_graph = tf.Graph()
    with my_graph.as_default():
        dequeue_ops = []
        for i in range(NUM_GPUS):
            with tf.name_scope('Tower%d' % i):
                input_placeholder, output_placeholder, epoch_ended_placeholder = queue_input_placeholders()
                gpu_queue = tf.FIFOQueue(
                    GPU_QUEUES_CAPACITY, [tf.float32, tf.float32, tf.bool], name='InputQueue{}'.format(i))
                
                enqueue_op = gpu_queue.enqueue(
                    [input_placeholder, output_placeholder, epoch_ended_placeholder])
                tf.add_to_collection('enqueue', enqueue_op)
                close_op = gpu_queue.close(cancel_pending_enqueues=True)
                tf.add_to_collection('close_queue', close_op)
                dequeue_ops.append(gpu_queue.dequeue())
    
    with tf.Session(graph=my_graph) as sess:
        sess.run(tf.global_variables_initializer())
        # data_provider.current_batch = 200
        threads = [EnqueueThread(data_provider, my_graph, sess, NUM_GPUS, EXAMPLES_PER_GPU) for _ in range(NUM_DATA_THREADS)]
        for t in threads:
            t.start()
        
        for i in range(60):
            before = time.time()
            dequeued = sess.run(dequeue_ops)
            print(dequeued[0][2], '---------------------------------------------------------------')
            print('DEQUEUEING took:', time.time() - before)
            time.sleep(1)
            
        EnqueueThread.coord.request_stop()
        sess.run(my_graph.get_collection('close_queue'))
        EnqueueThread.coord.join(threads)
