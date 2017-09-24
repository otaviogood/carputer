import tensorflow as tf
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

def weight_variable(shape, fanIn, fanOut, name=None, coll=None):
    with tf.variable_scope(coll):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(fanIn/2.0)) )


def weight_variable_c(shape, name=None, coll=None):
    fanIn = shape[0] * shape[1] * shape[2]
    fanOut = shape[0] * shape[1] * shape[3]
    with tf.variable_scope(coll):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(fanIn / 2.0)))


def bias_variable(shape, name=None, coll=None):
    with tf.variable_scope(coll):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# https://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
def flatten_batch(tensor):
    shape = tensor.get_shape().as_list()  # a list: [None, 9, 2]
    dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
    return tf.reshape(tensor, [-1, dim])  # -1 means "all"


class NNModel:
    # static class vars
    width = 128
    height = 128
    img_channels = 3

    def __init__(self):
        self.l2_collection = []

        # Set up the inputs to the conv net
        self.in_image = tf.placeholder(tf.float32, shape=[None, NNModel.width * NNModel.height * NNModel.img_channels], name='in_image')
        self.in_speed = tf.placeholder(tf.float32, shape=[None], name='in_speed')
        # Labels
        self.steering_regress_ = tf.placeholder(tf.float32, shape=[None], name='steering_regress_')
        self.throttle_regress_ = tf.placeholder(tf.float32, shape=[None], name='throttle_regress_')
        # misc
        self.keep_prob = tf.placeholder(tf.float32)
        self.train_mode = tf.placeholder(tf.float32)

        # Reshape and put input image in range [-0.5..0.5]
        x_image = tf.reshape(self.in_image, [-1, NNModel.width, NNModel.height, NNModel.img_channels])  / 255.0 - 0.5

        # Neural net layers - first convolution, then fully connected, then final transform to output
        act = self.conv_layer(x_image, 8, 5, 'conv1', 'shared_conv')
        act = self.conv_layer(act, 12, 5, 'conv2', 'shared_conv')
        act = self.conv_layer(act, 16, 5, 'conv3', 'shared_conv')
        act = self.conv_layer(act, 32, 5, 'conv4', 'shared_conv')

        act_flat = flatten_batch(act)
        act = self.fc_layer(act_flat, 1024, 'fc1', 'shared_fc', True)

        # -------------------- Insert discriminator here for domain adaptation --------------------
        # Sneak the speedometer value into the matrix
        in_speed_shaped = tf.reshape(self.in_speed, [-1, 1])
        act_concat = tf.concat([act, in_speed_shaped], 1)

        fc2_num_outs = 1024
        act = self.fc_layer(act_concat, fc2_num_outs, 'fc2', 'main', False)

        # Final transform without relu. Not doing l2 regularization on this because it just feels wrong.
        num_outputs = 1 + 1
        W_fc_final = weight_variable([fc2_num_outs, num_outputs], fc2_num_outs, num_outputs, name='W_fc4', coll='main')
        # self.l2_collection.append(W_fc4)
        b_fc_final = bias_variable([num_outputs], name='b_fc4', coll='main')
        final_activations = tf.matmul(act, W_fc_final) + b_fc_final

        # pick apart the final output matrix into steering and throttle continuous values.
        slice_a = 1
        self.steering_regress_result = tf.reshape(final_activations[:, 0:slice_a], [-1])
        slice_b = slice_a + 1
        self.throttle_regress_result = tf.reshape(final_activations[:, slice_a:slice_b], [-1])

        # we will optimized to minimize mean squared error
        self.squared_diff = tf.reduce_mean(tf.squared_difference(self.steering_regress_result, self.steering_regress_))
        self.squared_diff_throttle = tf.reduce_mean(tf.squared_difference(self.throttle_regress_result, self.throttle_regress_))

        # L2 regularization
        self.regularizers = sum([tf.nn.l2_loss(tensor) for tensor in self.l2_collection])

        # Add regression loss to regularizers. Arbitrary scalars to balance out the 3 things and give regression steering priority
        self.loss = 0.001 * self.regularizers + self.squared_diff*0.1 + self.squared_diff_throttle*5.0
        tf.summary.scalar('loss', self.loss)
        # -----------------------------------------------------------------------------------

        # http://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables
        optimizerA = tf.train.AdamOptimizer(3e-4)
        main_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_fc") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_conv") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "main")
        self.train_step = optimizerA.minimize(self.loss, var_list=main_train_vars)


    def conv_layer(self, tensor, channels_out, conv_size, name, scope_name):
        channels_in = tensor.get_shape().as_list()[3]
        W_conv = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='W_' + name, coll=scope_name)
        self.l2_collection.append(W_conv)
        b_conv = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        h_conv = tf.nn.relu(conv2d(tensor, W_conv) + b_conv)
        return max_pool_2x2(h_conv)


    def fc_layer(self, tensor, channels_out, name, scope_name, dropout):
        channels_in = tensor.get_shape().as_list()[1]
        W_fc = weight_variable([channels_in, channels_out], channels_in, channels_out, name='W_' + name, coll=scope_name)
        self.l2_collection.append(W_fc)
        b_fc = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        h_fc = tf.nn.relu(tf.matmul(tensor, W_fc) + b_fc)
        if dropout:
            h_fc = tf.nn.dropout(h_fc, self.keep_prob)
        return h_fc
