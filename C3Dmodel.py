import tensorflow as tf
from tensorflow import layers


def add_activation_summary(layer):
    # changes needed for multi cpu training!
    # i.e. remove "tower[0-9]*" from it!
    # tensor_name = layer.op.name
    # tf.summary.scalar("sparsity", tf.nn.zero_fraction(layer))
    # tf.summary.histogram("activations", layer)
    pass
    
    
def model_variable(name, shape, wd):
    def var_in_scope(scope):
        with tf.device('/cpu:0'), tf.variable_scope(scope):
            var = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(0,0.1))
        return var
    return var_in_scope


def inference(vid_placeholder, batch_size, dropout_rate, is_training, num_classes, collection):
    # dropout_rate = 1 - keep_rate !!!
    
    def conv3d(x, w, names):
        """
        conv3d inputs and filters
        input: [batch, in_depth, in_height, in_width, in_channels].
        filter: [filter_depth, filter_height, filter_width, in_channels, out_channels].
        """
        assert(len(names) == 3)
        conv = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=names[0])
        conv_bn = layers.batch_normalization(conv, 4, training=is_training, name=names[2])
        # conv_out = tf.nn.bias_add(conv, b)
        return tf.nn.elu(conv_bn, name=names[1])
    
    def max_pool(input, temp_k, name):
        # temp_k denotes kernel size and stride in temporal direction
        pool_out = tf.nn.max_pool3d(input, ksize=[1, temp_k, 2, 2, 1],
                                           strides=[1, temp_k, 2, 2, 1],
                                           padding='SAME', name=name)
        return pool_out

    weight_dict = {
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
        'wout': model_variable('wout', [4096, num_classes], 0.0005)
    }

    bias_dict = {
        # 'bconv1': model_variable('bconv1', [64], 0.000),
        # 'bconv2': model_variable('bconv2', [128], 0.000),
        # 'bconv3a': model_variable('bconv3a', [256], 0.000),
        # 'bconv3b': model_variable('bconv3b', [256], 0.000),
        # 'bconv4a': model_variable('bconv4a', [512], 0.000),
        # 'bconv4b': model_variable('bconv4b', [512], 0.000),
        # 'bconv5a': model_variable('bconv5a', [512], 0.000),
        # 'bconv5b': model_variable('bconv5b', [512], 0.000),
        # 'bfully1': model_variable('bfully1', [4096], 0.000),
        # 'bfully2': model_variable('bfully2', [4096], 0.000),
        'bout': model_variable('bout', [num_classes], 0.000),
    }
    
    with tf.variable_scope('convlayer1') as scope:
        # conv1 = conv3d(vid_placeholder, weight_dict['wconv1'](scope), bias_dict['bconv1'](scope), ('conv1', 'relu1', 'bn1'))
        conv1 = conv3d(vid_placeholder, weight_dict['wconv1'](scope), ('conv1', 'relu1', 'bn1'))
        add_activation_summary(conv1)
        pool1 = max_pool(conv1, 1, 'pool1')
        # print(type(pool1))    -> <class 'tensorflow.python.framework.ops.Tensor'>
        # print(type(pool1.op)) -> <class 'tensorflow.python.framework.ops.Operation'>
        # print(pool1.op.name)  -> convlayer1/pool1
        # print(pool1.name)     -> convlayer1/pool1:0
        # print(pool1.shape)    -> (1, 16, 56, 56, 64)
        # print(conv1.shape)    -> (1, 16, 112, 112, 64)
    
    with tf.variable_scope('convlayer2') as scope:
        conv2 = conv3d(pool1, weight_dict['wconv2'](scope), ('conv2', 'relu2', 'bn2'))
        # conv2 = conv3d(pool1, weight_dict['wconv2'](scope), bias_dict['bconv2'](scope), ('conv2', 'relu2', 'bn2'))
        # add_activation_summary(conv2)
        pool2 = max_pool(conv2, 2, 'pool2')
        
    with tf.variable_scope('convlayer3') as scope:
        conv3 = conv3d(pool2, weight_dict['wconv3a'](scope), ('conv3a', 'relu3a', 'bn3a'))
        # conv3 = conv3d(pool2, weight_dict['wconv3a'](scope), bias_dict['bconv3a'](scope), ('conv3a', 'relu3a', 'bn3a'))
        # add_activation_summary(conv3)
        conv3 = conv3d(conv3, weight_dict['wconv3b'](scope), ('conv3b', 'relu3b', 'bn3b'))
        # conv3 = conv3d(conv3, weight_dict['wconv3b'](scope), bias_dict['bconv3b'](scope), ('conv3b', 'relu3b', 'bn3b'))
        # add_activation_summary(conv3)
        pool3 = max_pool(conv3, 2, 'pool3')
        
    with tf.variable_scope('convlayer4') as scope:
        conv4 = conv3d(pool3, weight_dict['wconv4a'](scope), ('conv4a', 'relu4a', 'bn4a'))
        # conv4 = conv3d(pool3, weight_dict['wconv4a'](scope), bias_dict['bconv4a'](scope), ('conv4a', 'relu4a', 'bn4a'))
        # add_activation_summary(conv4)
        conv4 = conv3d(conv4, weight_dict['wconv4b'](scope), ('conv4b', 'relu4b', 'bn4b'))
        # conv4 = conv3d(conv4, weight_dict['wconv4b'](scope), bias_dict['bconv4b'](scope), ('conv4b', 'relu4b', 'bn4b'))
        # add_activation_summary(conv4)
        pool4 = max_pool(conv4, 2, 'pool4')
    
    with tf.variable_scope('convlayer5') as scope:
        conv5 = conv3d(pool4, weight_dict['wconv5a'](scope), ('conv5a', 'relu5a', 'bn5a'))
        # conv5 = conv3d(pool4, weight_dict['wconv5a'](scope), bias_dict['bconv5a'](scope), ('conv5a', 'relu5a', 'bn5a'))
        # add_activation_summary(conv5)
        conv5 = conv3d(conv5, weight_dict['wconv5b'](scope), ('conv5b', 'relu5b', 'bn5b'))
        # conv5 = conv3d(conv5, weight_dict['wconv5b'](scope), bias_dict['bconv5b'](scope), ('conv5b', 'relu5b', 'bn5b'))
        # add_activation_summary(conv5)
        pool5 = max_pool(conv5, 2, 'pool5')
        
    with tf.variable_scope('fully1') as scope:
        fully1_weights = weight_dict['wfully1'](scope)
        
    with tf.variable_scope('theFlattening'):
        pool5_flat = tf.reshape(pool5, [batch_size, fully1_weights.get_shape().as_list()[0]])

    with tf.variable_scope(scope):
        fully1 = tf.matmul(pool5_flat, fully1_weights)  # + bias_dict['bfully1'](scope)
        fully1 = layers.batch_normalization(fully1, 1, training=is_training, name='bn_fully1')
        fully1 = tf.nn.elu(fully1, name='relu_fully1')
        # add_activation_summary(fully1)
        # fully1 = tf.nn.dropout(fully1, dropout_rate)
    
    with tf.variable_scope('fully2') as scope:
        fully2 = tf.matmul(fully1, weight_dict['wfully2'](scope))  # + bias_dict['bfully2'](scope)
        fully2 = layers.batch_normalization(fully2, 1, training=is_training, name='bn_fully2')
        fully2 = tf.nn.elu(fully2, name='relu_fully2')
        # add_activation_summary(fully2)
        # fully2 = tf.nn.dropout(fully2, dropout_rate)
        
    with tf.variable_scope('out') as scope:
        out = tf.matmul(fully2, weight_dict['wout'](scope)) + bias_dict['bout'](scope)
        add_activation_summary(out)
    tf.add_to_collection(collection, out)
    return out
    
    
def loss(network_output, true_labels, collection):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=network_output, name='xentropy'))
    tf.summary.scalar('loss', loss)
    tf.add_to_collection(collection, loss)
    return loss


def train(loss, learning_rate, global_step, collection):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    tf.add_to_collection(collection, train_step)
    return train_step


def accuracy(net_output, onehot_labels, collection):
    correctly_classified = tf.equal(tf.argmax(net_output, 1), tf.argmax(onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctly_classified, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    tf.add_to_collection(collection, accuracy)
    tf.add_to_collection(collection, accuracy_summary)
    return accuracy, accuracy_summary
