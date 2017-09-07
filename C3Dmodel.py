import tensorflow as tf
from tensorflow import layers

WEIGHT_DECAY = 1e-05
INIT_STDDEV = 0.02

def add_activation_summary(layer):
    # changes needed for multi cpu training!
    # i.e. remove "tower[0-9]*" from it!
    # tensor_name = layer.op.name
    # tf.summary.scalar("sparsity", tf.nn.zero_fraction(layer))
    tf.summary.histogram("activations", layer)

def model_variable(name, shape, initializer, wd=None):
    def variable_creation_function():
        with tf.device('/cpu:0'):
            var = tf.get_variable(
                name, shape, dtype=tf.float32, initializer=initializer)
            if wd is not None:
                weight_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_loss)
        return var
    return variable_creation_function


def inference(vid_placeholder, batch_size, dropout_rate, is_training, num_classes, collection):
    # dropout_rate = 1 - keep_rate !!!
    
    def conv3d(x, w, b, names):
        """
        conv3d inputs and filters
        input: [batch, in_depth, in_height, in_width, in_channels].
        filter: [filter_depth, filter_height, filter_width, in_channels, out_channels].
        """
        conv = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=names[0])
        # conv_bn = layers.batch_normalization(
        #     conv, 4, training=is_training, name=names[2],
        #     beta_regularizer=tf.nn.l2_loss, gamma_regularizer=tf.nn.l2_loss)
        conv_out = tf.nn.bias_add(conv, b)
        return tf.nn.elu(conv_out, name=names[1])
    
    def max_pool(input, temp_k, name):
        # temp_k denotes kernel size and stride in temporal direction
        pool_out = tf.nn.max_pool3d(input, ksize=[1, temp_k, 2, 2, 1],
                                           strides=[1, temp_k, 2, 2, 1],
                                           padding='SAME', name=name)
        return pool_out

    FULLY1_DIM = 8192
    weight_init = tf.truncated_normal_initializer(0, INIT_STDDEV)
    # weight_init = tf.contrib.layers.xavier_initializer()
    weight_dict = {
        'wconv1': model_variable('wconv1', [3, 3, 3, 3, 64], weight_init, WEIGHT_DECAY),
        'wconv2': model_variable('wconv2', [3, 3, 3, 64, 128], weight_init, WEIGHT_DECAY),
        'wconv3a': model_variable('wconv3a', [3, 3, 3, 128, 256], weight_init, WEIGHT_DECAY),
        'wconv3b': model_variable('wconv3b', [3, 3, 3, 256, 256], weight_init, WEIGHT_DECAY),
        'wconv4a': model_variable('wconv4a', [3, 3, 3, 256, 512], weight_init, WEIGHT_DECAY),
        'wconv4b': model_variable('wconv4b', [3, 3, 3, 512, 512], weight_init, WEIGHT_DECAY),
        'wconv5a': model_variable('wconv5a', [3, 3, 3, 512, 512], weight_init, WEIGHT_DECAY),
        'wconv5b': model_variable('wconv5b', [3, 3, 3, 512, 512], weight_init, WEIGHT_DECAY),
        'wfully1': model_variable('wfully1', [FULLY1_DIM, 4096], weight_init, WEIGHT_DECAY),
        'wfully2': model_variable('wfully2', [4096, 4096], weight_init, WEIGHT_DECAY),
        'wout': model_variable('wout', [4096, num_classes], weight_init, WEIGHT_DECAY)
    }

    bias_init = tf.constant_initializer(0)
    bias_dict = {
        'bconv1': model_variable('bconv1', [64], bias_init),
        'bconv2': model_variable('bconv2', [128], bias_init),
        'bconv3a': model_variable('bconv3a', [256], bias_init),
        'bconv3b': model_variable('bconv3b', [256], bias_init),
        'bconv4a': model_variable('bconv4a', [512], bias_init),
        'bconv4b': model_variable('bconv4b', [512], bias_init),
        'bconv5a': model_variable('bconv5a', [512], bias_init),
        'bconv5b': model_variable('bconv5b', [512], bias_init),
        'bfully1': model_variable('bfully1', [4096], bias_init),
        'bfully2': model_variable('bfully2', [4096], bias_init),
        'bout': model_variable('bout', [num_classes], bias_init),
    }
    
    with tf.variable_scope('convlayer1'):
        conv1 = conv3d(vid_placeholder, weight_dict['wconv1'](), bias_dict['bconv1'](), ('conv1', 'relu1'))
        # conv1 = conv3d(vid_placeholder, weight_dict['wconv1'](), ('conv1', 'relu1', 'bn1'))
        # add_activation_summary(conv1)
        pool1 = max_pool(conv1, 1, 'pool1')
        # print(type(pool1))    -> <class 'tensorflow.python.framework.ops.Tensor'>
        # print(type(pool1.op)) -> <class 'tensorflow.python.framework.ops.Operation'>
        # print(pool1.op.name)  -> convlayer1/pool1
        # print(pool1.name)     -> convlayer1/pool1:0
        # print(pool1.shape)    -> (1, 16, 56, 56, 64)
        # print(conv1.shape)    -> (1, 16, 112, 112, 64)
    
    with tf.variable_scope('convlayer2'):
        # conv2 = conv3d(pool1, weight_dict['wconv2'](), ('conv2', 'relu2', 'bn2'))
        conv2 = conv3d(pool1, weight_dict['wconv2'](), bias_dict['bconv2'](), ('conv2', 'relu2'))
        # add_activation_summary(conv2)
        pool2 = max_pool(conv2, 2, 'pool2')
        
    with tf.variable_scope('convlayer3'):
        # conv3 = conv3d(pool2, weight_dict['wconv3a'](), ('conv3a', 'relu3a', 'bn3a'))
        conv3 = conv3d(pool2, weight_dict['wconv3a'](), bias_dict['bconv3a'](), ('conv3a', 'relu3a'))
        # add_activation_summary(conv3)
        
        # conv3 = conv3d(conv3, weight_dict['wconv3b'](), ('conv3b', 'relu3b', 'bn3b'))
        conv3 = conv3d(conv3, weight_dict['wconv3b'](), bias_dict['bconv3b'](), ('conv3b', 'relu3b'))
        # add_activation_summary(conv3)
        pool3 = max_pool(conv3, 2, 'pool3')
        
    with tf.variable_scope('convlayer4'):
        # conv4 = conv3d(pool3, weight_dict['wconv4a'](), ('conv4a', 'relu4a', 'bn4a'))
        conv4 = conv3d(pool3, weight_dict['wconv4a'](), bias_dict['bconv4a'](), ('conv4a', 'relu4a'))
        # add_activation_summary(conv4)
        
        # conv4 = conv3d(conv4, weight_dict['wconv4b'](), ('conv4b', 'relu4b', 'bn4b'))
        conv4 = conv3d(conv4, weight_dict['wconv4b'](), bias_dict['bconv4b'](), ('conv4b', 'relu4b'))
        # add_activation_summary(conv4)
        pool4 = max_pool(conv4, 2, 'pool4')
    
    with tf.variable_scope('convlayer5'):
        # conv5 = conv3d(pool4, weight_dict['wconv5a'](), ('conv5a', 'relu5a', 'bn5a'))
        conv5 = conv3d(pool4, weight_dict['wconv5a'](), bias_dict['bconv5a'](), ('conv5a', 'relu5a'))
        # add_activation_summary(conv5)
        # conv5 = conv3d(conv5, weight_dict['wconv5b'](), ('conv5b', 'relu5b', 'bn5b'))
        conv5 = conv3d(conv5, weight_dict['wconv5b'](), bias_dict['bconv5b'](), ('conv5b', 'relu5b'))
        # add_activation_summary(conv5)
        pool5 = max_pool(conv5, 2, 'pool5')

    with tf.variable_scope('theFlattening'):
        pool5_flat = tf.reshape(pool5, [batch_size, FULLY1_DIM])

    with tf.variable_scope('fully1'):
        fully1 = tf.matmul(pool5_flat, weight_dict['wfully1']()) + bias_dict['bfully1']()
        # fully1 = layers.batch_normalization(fully1, 1, training=is_training, name='bn_fully1')
        fully1 = tf.nn.elu(fully1, name='relu_fully1')
        # add_activation_summary(fully1)
        fully1 = tf.nn.dropout(fully1, dropout_rate)
    
    with tf.variable_scope('fully2'):
        fully2 = tf.matmul(fully1, weight_dict['wfully2']()) + bias_dict['bfully2']()
        # fully2 = layers.batch_normalization(fully2, 1, training=is_training, name='bn_fully2')
        fully2 = tf.nn.elu(fully2, name='relu_fully2')
        # add_activation_summary(fully2)
        fully2 = tf.nn.dropout(fully2, dropout_rate)
        
    with tf.variable_scope('out'):
        out = tf.matmul(fully2, weight_dict['wout']()) + bias_dict['bout']()
        out = out - tf.expand_dims(tf.reduce_max(out,1),1)
        add_activation_summary(out)
    tf.add_to_collection(collection, out)
    return out
    
    
def loss(network_output, training_labels, collection, scope):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=training_labels, logits=network_output, name='xentropy_per_batch')
    mean_cross_entropy = tf.reduce_mean(cross_entropy, name='mean_xentropy')
    tf.summary.scalar('mean_cross_entropy_per_batch', mean_cross_entropy)
    regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
    total_loss = tf.add(mean_cross_entropy, regularization_losses)
    tf.summary.scalar('total_loss_with_l2_regularization', total_loss)
    tf.add_to_collection(collection, total_loss)
    return total_loss, regularization_losses


# def train(loss, learning_rate, global_step, collection):
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     train_step = optimizer.minimize(loss, global_step=global_step)
#     tf.add_to_collection(collection, train_step)
#     return train_step


def accuracy(network_output, onehot_labels, collection):
    correctly_classified = tf.equal(tf.argmax(network_output, 1), tf.argmax(onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctly_classified, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    tf.add_to_collection(collection, accuracy)
    tf.add_to_collection(collection, accuracy_summary)
    return accuracy, accuracy_summary
