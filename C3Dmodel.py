import tensorflow as tf


def inference(vid_placeholder, batch_size, weight_dict, bias_dict, dropout_rate):
    # dropout_rate = 1 - keep_rate !!!!
    
    def conv3d(x, w, b, names):
        """
        conv3d inputs and filters
        input: [batch, in_depth, in_height, in_width, in_channels].
        filter: [filter_depth, filter_height, filter_width, in_channels, out_channels].
        """
        assert(len(names) == 2)
        conv = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=names[0])
        conv_out = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv_out, name=names[1])
    
    def max_pool(input, temp_k, name):
        # temp_k denotes kernel size and stride in temporal direction
        pool_out = tf.nn.max_pool3d(input, ksize=[1, temp_k, 2, 2, 1],
                                           strides=[1, temp_k, 2, 2, 1],
                                           padding='SAME', name=name)
        return pool_out
    
    with tf.name_scope('convlayer1'):
        conv1 = conv3d(vid_placeholder, weight_dict['wconv1'], bias_dict['bconv1'], ('conv1', 'relu1'))
        pool1 = max_pool(conv1, 1, 'pool1')
    
    with tf.name_scope('convlayer2'):
        conv2 = conv3d(pool1, weight_dict['wconv2'], bias_dict['bconv2'], ('conv2', 'relu2'))
        pool2 = max_pool(conv2, 2, 'pool2')
        
    with tf.name_scope('convlayer3'):
        conv3 = conv3d(pool2, weight_dict['wconv3a'], bias_dict['bconv3a'], ('conv3a', 'relu3a'))
        conv3 = conv3d(conv3, weight_dict['wconv3b'], bias_dict['bconv3b'], ('conv3b', 'relu3b'))
        pool3 = max_pool(conv3, 2, 'pool3')
        
    with tf.name_scope('convlayer4'):
        conv4 = conv3d(pool3, weight_dict['wconv4a'], bias_dict['bconv4a'], ('conv4a', 'relu4a'))
        conv4 = conv3d(conv4, weight_dict['wconv4b'], bias_dict['bconv4b'], ('conv4b', 'relu4b'))
        pool4 = max_pool(conv4, 2, 'pool4')
    
    with tf.name_scope('convlayer5'):
        conv5 = conv3d(pool4, weight_dict['wconv5a'], bias_dict['bconv5a'], ('conv5a', 'relu5a'))
        conv5 = conv3d(conv5, weight_dict['wconv5b'], bias_dict['bconv5b'], ('conv5b', 'relu5b'))
        pool5 = max_pool(conv5, 2, 'pool5')
        
    with tf.name_scope('theFlattening'):
        pool5_flat = tf.reshape(pool5, [batch_size, weight_dict['wfully1'].get_shape().as_list()[0]])

    with tf.name_scope('fully1'):
        fully1 = tf.matmul(pool5_flat, weight_dict['wfully1']) + bias_dict['bfully1']
        fully1 = tf.nn.relu(fully1, name='relu_fully1')
        fully1 = tf.nn.dropout(fully1, dropout_rate)
    
    with tf.name_scope('fully2'):
        fully2 = tf.matmul(fully1, weight_dict['wfully2']) + bias_dict['bfully2']
        fully2 = tf.nn.relu(fully2, name='relu_fully2')
        fully2 = tf.nn.dropout(fully2, dropout_rate)
        
    with tf.name_scope('out'):
        out = tf.matmul(fully2, weight_dict['wout']) + bias_dict['bout']
    return out
    
    
def loss(network_output, true_labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=network_output, name='xentropy'))
    return loss


def train(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step
