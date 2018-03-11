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


def weight_variable_c_const(shape, init_val, name=None, coll=None):
    with tf.variable_scope(coll):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(init_val))


def bias_variable(shape, init_val=0.0, name=None, coll=None):
    with tf.variable_scope(coll):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(init_val))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# experimental stuff that isn't working.
# returns [batch, height, width, channels]
def conv_superpool_2x2(x, in_channels, name, coll):
    # W_conv = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='W_' + name, coll=scope_name)
    # kernels = [[[[ 0.25,  0.25], [ 0.25,  0.25]],
    #             [[-0.25, -0.25], [ 0.25,  0.25]],
    #             [[-0.25,  0.25], [-0.25,  0.25]],
    #             [[-0.25,  0.25], [ 0.25, -0.25]]]]
    kernels = [[ [[ 0.25]], [[ 0.25]] ], [ [[ 0.25]], [[ 0.25]] ]]
    # kernels = [[ [[ 0.25]], [[ 0.25]] ], [ [[ 0.25]], [[ 0.25]] ],
    #            [ [[-0.25]], [[-0.25]] ], [ [[ 0.25]], [[ 0.25]] ],
    #            [ [[-0.25]], [[ 0.25]] ], [ [[-0.25]], [[ 0.25]] ],
    #            [ [[-0.25]], [[ 0.25]] ], [ [[ 0.25]], [[-0.25]] ]]
    kernels = np.array(kernels, dtype=np.float32)
    kernels = np.repeat(kernels, in_channels, axis=2)  # can I do this as a broadcast???
    b = bias_variable([in_channels], name='b_' + name, coll=coll)
    with tf.variable_scope(coll):
        v = tf.get_variable(name, initializer=tf.constant(kernels))
        # v = tf.constant(kernels, name=name)
    return tf.nn.depthwise_conv2d(x, v, strides=[1, 2, 2, 1], padding='VALID') + b


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Swish activation function: https://arxiv.org/pdf/1710.05941.pdf
def Swish(x, name=None, coll=None):
    channels_in = x.get_shape().as_list()[3]
    with tf.variable_scope(coll):
        beta = bias_variable([channels_in], init_val=1.0, name='b_' + name, coll=coll)
    return tf.nn.sigmoid(beta * x) * x


def LeakyTrainable(x, name=None, coll=None):
    channels_in = x.get_shape().as_list()[3]
    with tf.variable_scope(coll):
        beta = bias_variable([channels_in], init_val=0.0, name='b_' + name, coll=coll)
    return tf.maximum(x * beta, x)


# shake-shake regularization! I'm a fan. https://arxiv.org/abs/1705.07485
def ShakeShake(tensorA, tensorB, rand_shape, is_training):
    alpha = tf.random_uniform(rand_shape)  # [batch_size, 1, 1, channels_out]
    beta = tf.random_uniform(rand_shape)
    # tricky stuff to make the backwards pass have a different random than
    # the forward pass. Weirdest thing about shake-shake.
    shake_1 = beta * tensorA + tf.stop_gradient(alpha * tensorA - beta * tensorA)
    shake_2 = (1 - beta) * tensorB + tf.stop_gradient((1 - alpha) * tensorB - (1 - beta) * tensorB)
    def true_fun(): return shake_1 + shake_2
    def false_fun(): return (tensorA + tensorB) * 0.5
    return tf.cond(is_training, true_fun, false_fun)


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

        # ---- Neural net layers - first convolution, then fully connected, then final transform to output ----
        # act = self.conv_layer(x_image, 8, 5, 'conv1', 'shared_conv')
        act = avg_pool_2x2(x_image)  # downsample to 64x64 input image because we can.
        first, second = tf.split(act, 2, axis=1)  # cut off image above horizon. Now we are 64x32 only looking at ground.
        act = second

        # rand_noise = act + tf.random_normal(tf.shape(act))*0.1
        # act = tf.cond(self.is_training, lambda: rand_noise, lambda: act)

        self.visualizations["down_image"] = ("rgb_batch_steps", act)
        # act = avg_pool_2x2(act)
        # act = avg_pool_2x2(act)
        act = self.conv_layer_shake(act, 12, 5, 'conv2', 'shared_conv', visualize=True)
        act = self.conv_layer_shake(act, 16, 5, 'conv3', 'shared_conv')
        act = self.conv_layer_shake(act, 64, 5, 'conv4', 'shared_conv')

        # act = self.conv_layer(act, 12, 5, 'conv2', 'shared_conv')
        # act = self.conv_layer(act, 16, 5, 'conv3', 'shared_conv')
        # act = self.conv_layer(act, 32, 5, 'conv4', 'shared_conv')

        # act = self.conv_layer_shake(act, 12, 3, 'conv2a', 'shared_conv', do_pool=False)
        # act = self.conv_layer_shake(act, 12, 3, 'conv2b', 'shared_conv', do_pool=True)
        # act = self.conv_layer_shake(act, 16, 3, 'conv3a', 'shared_conv', do_pool=False)
        # act = self.conv_layer_shake(act, 16, 3, 'conv3b', 'shared_conv', do_pool=True)
        # act = self.conv_layer_shake(act, 32, 3, 'conv4a', 'shared_conv', do_pool=False)
        # act = self.conv_layer_shake(act, 32, 3, 'conv4b', 'shared_conv', do_pool=True)

        act_flat = flatten_batch(act)
        act = self.fc_layer(act_flat, 128, 'fc1', 'shared_fc', True, do_batch_norm=False)
        # actA = self.fc_layer(act_flat, 128, 'fc1A', 'shared_fc', True, do_batch_norm=False)
        # actB = self.fc_layer(act_flat, 128, 'fc1B', 'shared_fc', True, do_batch_norm=False)
        # act = ShakeShake(actA, actB, tf.shape(actA), self.is_training)

        # -------------------- Insert discriminator here for domain adaptation --------------------
        # Sneak the speedometer value into the matrix
        in_speed_shaped = tf.reshape(self.in_speed, [-1, 1])
        act_concat = tf.concat([act, in_speed_shaped], 1)

        fc2_num_outs = 256
        act = self.fc_layer(act_concat, fc2_num_outs, 'fc2', 'main', False, do_batch_norm=False)
        # actA = self.fc_layer(act_concat, fc2_num_outs, 'fc2A', 'main', False, do_batch_norm=False)
        # actB = self.fc_layer(act_concat, fc2_num_outs, 'fc2B', 'main', False, do_batch_norm=False)
        # act = ShakeShake(actA, actB, tf.shape(actA), self.is_training)

        # Final transform without relu. Not doing l2 regularization on this because it just feels wrong.
        num_outputs = 1 + 1
        final_activations = self.fc_layer(act, num_outputs, 'fc4', 'main', False, do_batch_norm=False, activation=False)
        # final_activationsA = self.fc_layer(act, num_outputs, 'fc4A', 'main', False, do_batch_norm=False, activation=False)
        # final_activationsB = self.fc_layer(act, num_outputs, 'fc4B', 'main', False, do_batch_norm=False, activation=False)
        # final_activations = ShakeShake(final_activationsA, final_activationsB, tf.shape(final_activationsA), self.is_training)

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
        # self.loss = 0.001 * self.regularizers + self.squared_diff*0.5 + self.squared_diff_throttle*1.0
        self.loss = self.squared_diff*0.5 + self.squared_diff_throttle*1.0
        tf.summary.scalar('loss', self.loss)
        # -----------------------------------------------------------------------------------

        # http://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables
        optimizerA = tf.train.AdamOptimizer(3e-4)
        main_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_fc") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_conv") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "main")
        self.train_step = optimizerA.minimize(self.loss)#, var_list=main_train_vars)


    def conv_layer(self, tensor, channels_out, conv_size, name, scope_name, do_pool = True, visualize=False):
        channels_in = tensor.get_shape().as_list()[3]
        W_conv = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='W_' + name, coll=scope_name)
        self.l2_collection.append(W_conv)
        b_conv = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        trans = conv2d(tensor, W_conv) + b_conv  # don't need the bias if batch norm is working.
        normed = batch_norm(name, trans, channels_out, self.is_training, convolutional=True, scope='BN')
        h_conv = tf.nn.relu(normed)
        if do_pool:
            h_conv = avg_pool_2x2(h_conv)
        if visualize:
            self.visualizations[name + "_viz"] = ("conv_conv_in_out", W_conv)
        return h_conv


    def conv_layer_shake(self, tensor, channels_out, conv_size, name, scope_name, do_pool = True, visualize=False):
        batch_size = tf.shape(tensor)[0]
        channels_in = tensor.get_shape().as_list()[3]
        W_convA = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='WA_' + name, coll=scope_name)
        W_convB = weight_variable_c([conv_size, conv_size, channels_in, channels_out], name='WB_' + name, coll=scope_name)
        # W_convA = weight_variable_c_const([conv_size, conv_size, channels_in, channels_out], 0.0, name='WA_' + name, coll=scope_name)
        # W_convB = weight_variable_c_const([conv_size, conv_size, channels_in, channels_out], 0.0, name='WB_' + name, coll=scope_name)
        b_convA = bias_variable([channels_out], name='bA_' + name, coll=scope_name)
        b_convB = bias_variable([channels_out], name='bB_' + name, coll=scope_name)
        transA = conv2d(tensor, W_convA) + b_convA  # don't need the bias if batch norm is working.
        transB = conv2d(tensor, W_convB) + b_convB  # don't need the bias if batch norm is working.
        normedA = batch_norm(name, transA, channels_out, self.is_training, convolutional=True, scope='BNA')
        normedB = batch_norm(name, transB, channels_out, self.is_training, convolutional=True, scope='BNB')
        # h_convA = tf.nn.relu(transA)
        # h_convB = tf.nn.relu(transB)
        h_convA = tf.nn.relu(normedA)
        h_convB = tf.nn.relu(normedB)
        # h_convA = LeakyTrainable(normedA, name='SwishA_' + name, coll=scope_name)
        # h_convB = LeakyTrainable(normedB, name='SwishB_' + name, coll=scope_name)
        # h_convA = tf.concat([tf.nn.relu(normedA), tf.minimum(0.0, normedA)], 3)  # concatenated relus
        # h_convB = tf.concat([tf.nn.relu(normedB), tf.minimum(0.0, normedB)], 3)

        h_conv = ShakeShake(h_convA, h_convB, tf.shape(h_convA), self.is_training)

        if do_pool:
            h_conv = avg_pool_2x2(h_conv)
            # h_conv = tf.nn.relu(conv_superpool_2x2(h_conv, channels_out, "superpool_" + name, coll=scope_name))
        if visualize:
            self.visualizations[name + "_vizA"] = ("conv_conv_in_out", W_convA)
            self.visualizations[name + "_vizB"] = ("conv_conv_in_out", W_convB)
        return h_conv


    def fc_layer(self, tensor, channels_out, name, scope_name, dropout, do_batch_norm = False, activation = tf.nn.relu):
        channels_in = tensor.get_shape().as_list()[1]
        W_fc = weight_variable([channels_in, channels_out], channels_in, channels_out, name='W_' + name, coll=scope_name)
        self.l2_collection.append(W_fc)
        b_fc = bias_variable([channels_out], name='b_' + name, coll=scope_name)
        trans = tf.matmul(tensor, W_fc) + b_fc  # don't need the bias if batch norm is working.
        if do_batch_norm:
            trans = batch_norm(name, trans, channels_out, self.is_training, convolutional=False, scope='BN')
        if activation:
            h_fc = activation(trans)
        else:
            h_fc = trans
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
        self.loss = 0.001 * self.regularizers + self.squared_diff*0.5 + self.squared_diff_throttle*1.0

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
