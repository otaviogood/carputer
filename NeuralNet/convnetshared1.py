import tensorflow as tf
from tensorflow.contrib import rnn
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

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# https://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
def flatten_batch(tensor):
    shape = tensor.get_shape().as_list()  # a list: [None, 9, 2]
    dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
    return tf.reshape(tensor, [-1, dim])  # -1 means "all"


def batch_norm(name, x, n_out, phase_train, convolutional=False, scope='BN'):
    """
    Args:
        x:           Tensor
        n_out:       integer, depth of input maps
        phase_train: tf.bool, true indicates training phase
        convolutional:tf.bool, true is conv layer, false is fully connected layer
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    # n_out = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope+name):
        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float32)
        beta = tf.get_variable(name='beta'+name, dtype=tf.float32, initializer=init_beta, regularizer=None, trainable=True)
        gamma = tf.get_variable(name='gamma'+name, dtype=tf.float32, initializer=init_gamma, regularizer=None, trainable=True)
        if convolutional:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0])
        ema = tf.train.ExponentialMovingAverage(decay=0.995)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

class NNModel:
    # static class vars

    n_steps = 1  # timesteps for RNN-like training (no RNN here. This is the Alexnet.)
    n_hidden = 128  # hidden layer num of features
    # n_vocab = 256

    def __init__(self):
        self.l2_collection = []
        self.visualizations = {}

        # Set up the inputs to the conv net
        self.in_image = tf.placeholder(tf.float32, shape=[None, config.width * config.height * config.img_channels], name='in_image')
        self.in_image_small = tf.placeholder(tf.float32, shape=[None, config.width_small * config.height_small * config.img_channels], name='in_image_small')
        self.in_speed = tf.placeholder(tf.float32, shape=[None], name='in_speed')
        # Labels
        self.steering_regress_ = tf.placeholder(tf.float32, shape=[None], name='steering_regress_')
        self.throttle_regress_ = tf.placeholder(tf.float32, shape=[None], name='throttle_regress_')
        # misc
        self.keep_prob = tf.placeholder(tf.float32)
        self.train_mode = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        # Reshape and put input image in range [-0.5..0.5]
        x_image = tf.reshape(self.in_image, [-1, config.width, config.height, config.img_channels]) / 255.0 - 0.5

        # Neural net layers - first convolution, then fully connected, then final transform to output
        # act = self.conv_layer(x_image, 8, 5, 'conv1', 'shared_conv')
        act = avg_pool_2x2(x_image)
        # act = avg_pool_2x2(act)
        # act = avg_pool_2x2(act)
        act = self.conv_layer(act, 12, 5, 'conv2', 'shared_conv')
        act = self.conv_layer(act, 16, 5, 'conv3', 'shared_conv')
        act = self.conv_layer(act, 32, 5, 'conv4', 'shared_conv')
        # act = self.conv_layer(act, 12, 3, 'conv2a', 'shared_conv', do_pool=False)
        # act = self.conv_layer(act, 12, 3, 'conv2b', 'shared_conv', do_pool=True)
        # act = self.conv_layer(act, 16, 3, 'conv3a', 'shared_conv', do_pool=False)
        # act = self.conv_layer(act, 16, 3, 'conv3b', 'shared_conv', do_pool=True)
        # act = self.conv_layer(act, 32, 3, 'conv4a', 'shared_conv', do_pool=False)
        # act = self.conv_layer(act, 32, 3, 'conv4b', 'shared_conv', do_pool=True)

        act_flat = flatten_batch(act)
        act = self.fc_layer(act_flat, 256, 'fc1', 'shared_fc', True, batch_norm=False)

        # -------------------- Insert discriminator here for domain adaptation --------------------
        # Sneak the speedometer value into the matrix
        in_speed_shaped = tf.reshape(self.in_speed, [-1, 1])
        act_concat = tf.concat([act, in_speed_shaped], 1)

        fc2_num_outs = 256
        act = self.fc_layer(act_concat, fc2_num_outs, 'fc2', 'main', False, batch_norm=False)

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


    def conv_layer(self, tensor, channels_out, conv_size, name, scope_name, do_pool = True):
        channels_in = tensor.get_shape().as_list()[3]
        W_conv = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='W_' + name, coll=scope_name)
        self.l2_collection.append(W_conv)
        b_conv = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        trans = conv2d(tensor, W_conv) + b_conv  # don't need the bias if batch norm is working.
        normed = batch_norm(name, trans, channels_out, self.is_training, convolutional=True, scope='BN')
        h_conv = tf.nn.relu(normed)
        if do_pool:
            h_conv = avg_pool_2x2(h_conv)
        return h_conv


    def fc_layer(self, tensor, channels_out, name, scope_name, dropout, batch_norm = False):
        channels_in = tensor.get_shape().as_list()[1]
        W_fc = weight_variable([channels_in, channels_out], channels_in, channels_out, name='W_' + name, coll=scope_name)
        self.l2_collection.append(W_fc)
        b_fc = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        trans = tf.matmul(tensor, W_fc) + b_fc  # don't need the bias if batch norm is working.
        if batch_norm:
            trans = batch_norm(name, trans, channels_out, self.is_training, convolutional=False, scope='BN')
        h_fc = tf.nn.relu(trans)
        if dropout:
            h_fc = tf.nn.dropout(h_fc, self.keep_prob)
        return h_fc


class LSTMModel:
    # static class vars

    n_steps = 32  # timesteps for RNN-like training
    n_hidden = 64  # hidden layer num of features

    def __init__(self):
        self.l2_collection = []
        self.visualizations = {}

        # Set up the inputs to the conv net
        self.in_image = tf.placeholder(tf.float32, shape=[None, config.width * config.height * config.img_channels], name='in_image')
        self.in_image_small = tf.placeholder(tf.float32, shape=[None, self.n_steps, config.width_small * config.height_small * config.img_channels], name='in_image_small')
        # self.visualizations["in_image"] = ("rgb_batch_steps", tf.reshape() self.in_image)
        self.in_speed = tf.placeholder(tf.float32, shape=[None, self.n_steps, 1], name='in_speed')
        # Labels
        self.steering_regress_ = tf.placeholder(tf.float32, shape=[None, self.n_steps, 1], name='steering_regress_')
        self.throttle_regress_ = tf.placeholder(tf.float32, shape=[None, self.n_steps, 1], name='throttle_regress_')
        # misc
        self.keep_prob = tf.placeholder(tf.float32)
        self.train_mode = tf.placeholder(tf.float32)

        real_lstm = False
        if real_lstm:
            # Reshape and put input image in range [-0.5..0.5]
            x_image = tf.reshape(self.in_image_small, [-1, self.n_steps, config.width_small, config.height_small, config.img_channels]) / 255.0 - 0.5
            batch_size = tf.shape(x_image)[0]

            act = tf.reshape(x_image, [-1, config.width_small, config.height_small, config.img_channels])
            # act = avg_pool_2x2(act)

            self.visualizations["down_image"] = ("rgb_batch_steps", act)
            # act = self.conv_layer(act, 12, 5, 'conv2', 'shared_conv')
            # act = self.conv_layer(act, 4, 5, 'conv3', 'shared_conv')
            final_channels = 16
            act = self.conv_layer(act, final_channels, 5, 'conv4', 'shared_conv')

            new_width = act.get_shape().as_list()[1]

            act = flatten_batch(act)

            # Sneak the speedometer value into the matrix
            in_speed_shaped = tf.reshape(self.in_speed, [-1, 1]) * 10.0
            act = tf.concat([act, in_speed_shaped], 1)


            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_vocab)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_vocab)

            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, width, height, channels)
            act = tf.reshape(act, [-1, self.n_steps, new_width * new_width * final_channels + 1])
            act = tf.unstack(act, self.n_steps, 1)

            # Define a lstm cell with tensorflow
            with tf.variable_scope('lstm'):
                lstm_cell = rnn.BasicLSTMCell(self.n_hidden)
                # lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.n_hidden)

                # Get lstm cell output
                # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
                state = lstm_cell.zero_state(batch_size, tf.float32)
                outputs = []
                for input_ in act:
                    output, state = lstm_cell(input_, state)
                    outputs.append(output)
                # return (outputs, state)
            lstm_out = outputs[-1]
        else:
            act = tf.reshape(self.in_image_small / 255.0 - 0.5, [-1, self.n_steps, config.width_small, config.height_small, config.img_channels])
            act = tf.transpose(act, [0, 2, 3, 1, 4])
            act = tf.reshape(act, [-1, config.width_small, config.height_small, self.n_steps * config.img_channels])
            act = self.conv_layer(act, self.n_steps*4, 5, 'conv1c', 'shared_conv')
            act = flatten_batch(act)
            in_speed_shaped = tf.reshape(self.in_speed, [-1, self.n_steps]) * 10.0
            act = tf.concat([act, in_speed_shaped], 1)
            act = self.fc_layer(act, 128, 'fc1c', 'shared_fc', True)
            lstm_out = act

        # Reshape and put input image in range [-0.5..0.5]
        x_image2 = tf.reshape(self.in_image / 255.0 - 0.5, [-1, config.width, config.height, config.img_channels])
        # self.visualizations["pathB_image"] = ("rgb_batch", x_image2)

        # Neural net layers - first convolution, then fully connected, then final transform to output
        act = self.conv_layer(x_image2, 8, 5, 'conv1b', 'shared_conv')
        # act = max_pool_2x2(x_image2)
        act = self.conv_layer(act, 12, 5, 'conv2b', 'shared_conv')
        act = self.conv_layer(act, 16, 5, 'conv3b', 'shared_conv')
        act = self.conv_layer(act, 32, 5, 'conv4b', 'shared_conv')

        act = flatten_batch(act)
        self.mid_act = act
        self.mid_lstm = lstm_out * 0.01
        act = tf.concat([act, self.mid_lstm], 1)
        # act = self.mid_lstm
        act = self.fc_layer(act, 1024, 'fc1', 'shared_fc', True)

        # -------------------- Insert discriminator here for domain adaptation --------------------
        # Sneak the speedometer value into the matrix
        in_speed_last = tf.transpose(self.in_speed, [1, 0, 2])[-1]  # [batch_size, 1]
        # in_speed_shaped = tf.reshape(in_speed_last, [-1, 1])
        # act = tf.concat([act, in_speed_last, lstm_out], 1)
        act = tf.concat([act, in_speed_last], 1)

        fc2_num_outs = 1024
        act = self.fc_layer(act, fc2_num_outs, 'fc2', 'main', False)



        num_outputs = 2
        W_fc_final = weight_variable([fc2_num_outs, num_outputs], fc2_num_outs, num_outputs, name='W_fc4', coll='main')
        b_fc_final = bias_variable([num_outputs], name='b_fc4', coll='main')
        regress_outs = tf.matmul(act, W_fc_final) + b_fc_final

        self.steering_regress_result = tf.reshape(regress_outs[:, 0], [-1])  # [batch_size]
        self.throttle_regress_result = tf.reshape(regress_outs[:, 1], [-1])  # [batch_size]

        # we will optimized to minimize mean squared error
        steering_temp = tf.reshape(self.steering_regress_[:, -1], [-1])
        throttle_temp = tf.reshape(self.throttle_regress_[:, -1], [-1])
        # steering_regress_last = tf.transpose(self.steering_regress_result, [1, 0, 2])[-1]
        # throttle_regress_last = tf.transpose(self.throttle_regress_result, [1, 0, 2])[-1]
        self.squared_diff = tf.reduce_mean(tf.squared_difference(self.steering_regress_result, steering_temp))
        self.squared_diff_throttle = tf.reduce_mean(tf.squared_difference(self.throttle_regress_result, throttle_temp))

        self.regularizers = sum([tf.nn.l2_loss(tensor) for tensor in self.l2_collection])
        # self.regularizers = tf.zeros((1))
        # Add regression loss to regularizers. Arbitrary scalars to balance out the 3 things and give regression steering priority
        # self.loss = self.squared_diff * 0.1 + self.squared_diff_throttle * 5.0
        self.loss = 0.001 * self.regularizers + self.squared_diff*0.1 + self.squared_diff_throttle*5.0

        optimizerA = tf.train.AdamOptimizer(3e-4)
        self.train_step = optimizerA.minimize(self.loss)


    def conv_layer(self, tensor, channels_out, conv_size, name, scope_name, pool=True):
        channels_in = tensor.get_shape().as_list()[3]
        W_conv = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='W_' + name,
                                   coll=scope_name)
        self.l2_collection.append(W_conv)
        b_conv = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        h_conv = tf.nn.relu(conv2d(tensor, W_conv) + b_conv)
        if pool:
            return max_pool_2x2(h_conv)
        else:
            return h_conv

    def fc_layer(self, tensor, channels_out, name, scope_name, dropout):
        channels_in = tensor.get_shape().as_list()[1]
        W_fc = weight_variable([channels_in, channels_out], channels_in, channels_out, name='W_' + name,
                               coll=scope_name)
        self.l2_collection.append(W_fc)
        b_fc = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        h_fc = tf.nn.relu(tf.matmul(tensor, W_fc) + b_fc)
        if dropout:
            h_fc = tf.nn.dropout(h_fc, self.keep_prob)
        return h_fc
