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


class NNModel:
    # static class vars
    width = 128
    height = 128
    widthD2 = width / 2
    heightD2 = height / 2
    widthD4 = width / 4
    heightD4 = height / 4
    widthD8 = width / 8
    heightD8 = height / 8
    widthD16 = width / 16
    heightD16 = height / 16
    img_channels = 3
    fc1_num_outs = 1024
    fc2_num_outs = 1024
    l1_conv_size = 5
    l1_num_convs = 8  # 8
    l2_conv_size = 5
    l2_num_convs = 12  # 12
    l3_conv_size = 5
    l3_num_convs = 16  # 16
    l4_conv_size = 5
    l4_num_convs = 32  # 32

    def __init__(self):
        self.in_image = tf.placeholder(tf.float32, shape=[None, NNModel.width * NNModel.height * NNModel.img_channels], name='in_image')
        self.in_speed = tf.placeholder(tf.float32, shape=[None], name='in_speed')
        self.steering_regress_ = tf.placeholder(tf.float32, shape=[None], name='steering_regress_')
        self.throttle_regress_ = tf.placeholder(tf.float32, shape=[None], name='throttle_regress_')

        self.keep_prob = tf.placeholder(tf.float32)
        self.train_mode = tf.placeholder(tf.float32)

        W_conv1 = weight_variable_c([NNModel.l1_conv_size, NNModel.l1_conv_size, NNModel.img_channels, NNModel.l1_num_convs], name='W_conv1', coll='shared_conv')
        b_conv1 = bias_variable([NNModel.l1_num_convs], name='b_conv1', coll='shared_conv')
        x_image = tf.reshape(self.in_image, [-1, NNModel.width, NNModel.height, NNModel.img_channels])
        h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(x_image / 255.0 - 0.5, W_conv1) + b_conv1))
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable_c([NNModel.l2_conv_size, NNModel.l2_conv_size, NNModel.l1_num_convs, NNModel.l2_num_convs], name='W_conv2', coll='shared_conv')
        b_conv2 = bias_variable([NNModel.l2_num_convs], name='b_conv2', coll='shared_conv')
        h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_pool1, W_conv2) + b_conv2))
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable_c([NNModel.l3_conv_size, NNModel.l3_conv_size, NNModel.l2_num_convs, NNModel.l3_num_convs], name='W_conv3', coll='shared_conv')
        b_conv3 = bias_variable([NNModel.l3_num_convs], name='b_conv3', coll='shared_conv')
        h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_pool2, W_conv3) + b_conv3))
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable_c([NNModel.l4_conv_size, NNModel.l4_conv_size, NNModel.l3_num_convs, NNModel.l4_num_convs], name='W_conv4', coll='shared_fc')
        b_conv4 = bias_variable([NNModel.l4_num_convs], name='b_conv4', coll='shared_fc')
        h_conv4 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_pool3, W_conv4) + b_conv4))
        self.h_pool4 = max_pool_2x2(h_conv4)

        W_fc1 = weight_variable([NNModel.widthD16 * NNModel.heightD16 * NNModel.l4_num_convs, NNModel.fc1_num_outs], NNModel.widthD16 * NNModel.heightD16 * NNModel.l4_num_convs, NNModel.fc1_num_outs, name='W_fc1', coll='shared_fc')
        b_fc1 = bias_variable([NNModel.fc1_num_outs], name='b_fc1', coll='shared_fc')
        self.h_pool5_flat = tf.reshape(self.h_pool4, [-1, NNModel.widthD16 * NNModel.heightD16 * NNModel.l4_num_convs])
        # h_pool5_concat = tf.concat([self.h_pool5_flat, self.lat_, self.lon_], 1)

        h_fc1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.h_pool5_flat, W_fc1) + b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)


        # -------------------- Path for steer/throttle/lat/lon --------------------
        # W_fc2 = weight_variable([NNModel.fc1_num_outs + config.latlon_buckets * config.latlon_buckets + 1, NNModel.fc2_num_outs], NNModel.fc1_num_outs + config.latlon_buckets * config.latlon_buckets + 1, NNModel.fc2_num_outs, name='W_fc2', coll='main')
        W_fc2 = weight_variable([NNModel.fc1_num_outs + 1, NNModel.fc2_num_outs], NNModel.fc1_num_outs + 1, NNModel.fc2_num_outs, name='W_fc2', coll='main')
        b_fc2 = bias_variable([NNModel.fc2_num_outs], name='b_fc2', coll='main')
        # latlon_flat = tf.reshape(self.latlon_, [-1, config.latlon_buckets * config.latlon_buckets])
        in_speed_shaped = tf.reshape(self.in_speed, [-1, 1])
        # h_fc1_drop_concat = tf.concat([h_fc1_drop, latlon_flat*0.0, in_speed_shaped*0.0], 1)  # Kill the lat-lon input for now.
        h_fc1_drop_concat = tf.concat([h_fc1_drop, in_speed_shaped], 1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop_concat, W_fc2) + b_fc2)

        # num_outputs = 2 * NNModel.max_log_outs + 2 * config.latlon_buckets + 1 + config.latlon_buckets * config.latlon_buckets
        num_outputs = 1 + 1
        W_fc4 = weight_variable([NNModel.fc2_num_outs, num_outputs], NNModel.fc2_num_outs, num_outputs, name='W_fc4', coll='main')
        b_fc4 = bias_variable([num_outputs], name='b_fc4', coll='main')

        final_activations = tf.matmul(h_fc2, W_fc4) + b_fc4
        slice_a = 1
        self.steering_regress_result = tf.reshape(final_activations[:, 0:slice_a], [-1])
        slice_b = slice_a + 1
        self.throttle_regress_result = tf.reshape(final_activations[:, slice_a:slice_b], [-1])


        self.squared_diff = tf.reduce_mean(tf.squared_difference(self.steering_regress_result, self.steering_regress_))
        self.squared_diff_throttle = tf.reduce_mean(tf.squared_difference(self.throttle_regress_result, self.throttle_regress_))

        # L2 regularization
        self.regularizers = (tf.nn.l2_loss(W_conv1) +
                        tf.nn.l2_loss(W_conv2) +
                        tf.nn.l2_loss(W_conv3) +
                        tf.nn.l2_loss(W_conv4) +
                        tf.nn.l2_loss(W_fc1) +
                        tf.nn.l2_loss(W_fc2) +
                        tf.nn.l2_loss(W_fc4)
                        )
        # Add the regularization term to the loss.
        # Add regression loss to cross entropy loss and regularizers. Arbitrary scalars to balance out the 3 things and give regression steering priority
        # self.squared_diff = tf.reduce_mean(tf.squared_difference(self.steering_regress_result, self.steering_regress_))
        # self.squared_diff = tf.squared_difference(self.steering_regress_result, self.steering_regress_)
        self.loss = 0.001 * self.regularizers + self.squared_diff*0.1 + self.squared_diff_throttle*2.0
        tf.summary.scalar('loss', self.loss)
        # -----------------------------------------------------------------------------------


        # http://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables
        # ALEX-NET
        optimizerA = tf.train.AdamOptimizer(3e-4)
        main_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_fc") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_conv") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "main")
        self.train_step = optimizerA.minimize(self.loss, var_list=main_train_vars)
