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
    max_log_outs = 15
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
    fc1_num_outs = 256
    fc2_num_outs = 256
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
        self.steering_ = tf.placeholder(tf.float32, shape=[None, NNModel.max_log_outs], name='steering_')
        self.steering_regress_ = tf.placeholder(tf.float32, shape=[None], name='steering_regress_')
        self.throttle_regress_ = tf.placeholder(tf.float32, shape=[None], name='throttle_regress_')
        self.throttle_ = tf.placeholder(tf.float32, shape=[None, NNModel.max_log_outs], name='throttle_')
        # self.lat_ = tf.placeholder(tf.float32, shape=[None, config.latlon_buckets], name='lat_')
        # self.lon_ = tf.placeholder(tf.float32, shape=[None, config.latlon_buckets], name='lon_')

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
        num_outputs = 2 * NNModel.max_log_outs + 1 + 1
        W_fc4 = weight_variable([NNModel.fc2_num_outs, num_outputs], NNModel.fc2_num_outs, num_outputs, name='W_fc4', coll='main')
        b_fc4 = bias_variable([num_outputs], name='b_fc4', coll='main')

        final_activations = tf.matmul(h_fc2, W_fc4) + b_fc4
        slice_a = NNModel.max_log_outs
        self.steering_softmax = tf.nn.softmax(final_activations[:, :slice_a])
        slice_b = slice_a + NNModel.max_log_outs
        self.throttle_softmax = tf.nn.softmax(final_activations[:, slice_a:slice_b])
        # slice_c = slice_b + config.latlon_buckets
        # self.lat_softmax = tf.nn.softmax(final_activations[:, slice_b:slice_c])
        # slice_d = slice_c + config.latlon_buckets
        # self.lon_softmax = tf.nn.softmax(final_activations[:, slice_c:slice_d])
        slice_e = slice_b + 1
        self.steering_regress_result = tf.reshape(final_activations[:, slice_b:slice_e], [-1])
        slice_f = slice_e + 1
        self.throttle_regress_result = tf.reshape(final_activations[:, slice_e:slice_f], [-1])

        # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        # http://stackoverflow.com/questions/33712178/tensorflow-nan-bug

        steering_cross_entropy = -tf.reduce_mean(self.steering_ * tf.log(tf.clip_by_value(self.steering_softmax, 1e-10, 1.0)))
        throttle_cross_entropy = -tf.reduce_mean(self.throttle_ * tf.log(tf.clip_by_value(self.throttle_softmax, 1e-10, 1.0)))
        # lat_cross_entropy = -tf.reduce_mean(self.lat_ * tf.log(tf.clip_by_value(self.lat_softmax, 1e-10, 1.0)))
        # lon_cross_entropy = -tf.reduce_mean(self.lon_ * tf.log(tf.clip_by_value(self.lon_softmax, 1e-10, 1.0)))

        self.squared_diff = tf.reduce_mean(tf.squared_difference(self.steering_regress_result, self.steering_regress_))
        self.squared_diff_throttle = tf.reduce_mean(tf.squared_difference(self.throttle_regress_result, self.throttle_regress_))
        # cross_entropy = (steering_cross_entropy + (throttle_cross_entropy + lat_cross_entropy + lon_cross_entropy)) / 4.0
        # cross_entropy = (steering_cross_entropy + (throttle_cross_entropy + latlon_cross_entropy + lat_cross_entropy + lon_cross_entropy)) / 5.0
        cross_entropy = (throttle_cross_entropy + steering_cross_entropy) / 2.0
        tf.summary.scalar('cross_entropy', cross_entropy)

        # L2 regularization
        regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                        tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
                        tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
                        tf.nn.l2_loss(W_conv4) + tf.nn.l2_loss(b_conv4) +
                        tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                        tf.nn.l2_loss(W_fc4) + tf.nn.l2_loss(b_fc4)
                        )
        # Add the regularization term to the loss.
        # Add regression loss to cross entropy loss and regularizers. Arbitrary scalars to balance out the 3 things and give regression steering priority
        # self.squared_diff = tf.reduce_mean(tf.squared_difference(self.steering_regress_result, self.steering_regress_))
        # self.squared_diff = tf.squared_difference(self.steering_regress_result, self.steering_regress_)
        cross_entropy = cross_entropy*1.0 + 0.0001 * regularizers + self.squared_diff*0.1 + self.squared_diff_throttle*0.1
        # cross_entropy = cross_entropy*1.0 + 0.0001 * regularizers
        # -----------------------------------------------------------------------------------


        # http://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables
        # ALEX-NET
        # optimizerA = tf.train.MomentumOptimizer(1.0, 0.9)
        optimizerA = tf.train.AdamOptimizer(3e-4)
        main_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_fc") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "shared_conv") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "main")
        self.train_step = optimizerA.minimize(cross_entropy, var_list=main_train_vars)


        # THIS NOW DOES A SUM OF SQUARES, NOT AN EXACT MATCH.
        def compute_pred_accuracy(output, label):
            prediction = tf.argmax(output, 1)
            delta = (prediction - tf.argmax(label, 1))
            energy = delta * delta
            accuracy = tf.reduce_mean(tf.cast(energy, "float"))

            return prediction, accuracy

        self.steering_pred, self.steering_accuracy = compute_pred_accuracy(self.steering_softmax, self.steering_)
        self.throttle_pred, self.throttle_accuracy = compute_pred_accuracy(self.throttle_softmax, self.throttle_)
        # self.lat_pred, self.lat_accuracy = compute_pred_accuracy(self.lat_softmax, self.lat_)
        # self.lon_pred, self.lon_accuracy = compute_pred_accuracy(self.lon_softmax, self.lon_)



# width = 128
# height = 128
# widthD2 = width / 2
# heightD2 = height / 2
# widthD4 = width / 4
# heightD4 = height / 4
# widthD8 = width / 8
# heightD8 = height / 8
# widthD16 = width / 16
# heightD16 = height / 16
# max_log_outs = 15
# img_channels = 3
# fc1_num_outs = 256
# l1_conv_size = 3
# l1_num_convs = 8
# l2_conv_size = 3
# l2_num_convs = 12
# l3_conv_size = 3
# l3_num_convs = 16
# l4_conv_size = 3
# l4_num_convs = 32
#
# numPulses = 15
# pulseScale = 64.0 #, 120.0
#
# h_pool5_odo_concat = None
#
#
# def gen_graph_ops():
#     global h_pool5_odo_concat
#     x = tf.placeholder(tf.float32, shape=[None, width * height * img_channels], name='x')
#     odo = tf.placeholder(tf.float32, shape=[None, 1], name='odo')
#     vel = tf.placeholder(tf.float32, shape=[None, 1], name='vel')
#     pulse = tf.placeholder(tf.float32, shape=[None, numPulses], name='pulse')
#     steering_ = tf.placeholder(tf.float32, shape=[None, max_log_outs], name='steering_')
#     throttle_ = tf.placeholder(tf.float32, shape=[None, max_log_outs], name='throttle_')
#
#     keep_prob = tf.placeholder(tf.float32)
#     train_mode = tf.placeholder(tf.float32)
#
#     W_conv1 = weight_variable([l1_conv_size, l1_conv_size, img_channels, l1_num_convs], l1_conv_size * l1_conv_size * img_channels, l1_conv_size * l1_conv_size * l1_num_convs, name='W_conv1')
#     b_conv1 = bias_variable([l1_num_convs])
#     x_image = tf.reshape(x, [-1, width, height, img_channels])
#     h_conv1 = tf.nn.relu(conv2d(x_image / 255.0 - 0.5, W_conv1) + b_conv1)
#     h_pool1 = max_pool_2x2(h_conv1)#, name='h_pool1', collections='htmlize')
#
#     W_conv2 = weight_variable([l2_conv_size, l2_conv_size, l1_num_convs, l2_num_convs], l2_conv_size * l2_conv_size * l1_num_convs, l2_conv_size * l2_conv_size * l2_num_convs, name='W_conv2')
#     b_conv2 = bias_variable([l2_num_convs])
#     h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#     h_pool2 = max_pool_2x2(h_conv2)
#
#     W_conv3 = weight_variable([l3_conv_size, l3_conv_size, l2_num_convs, l3_num_convs], l3_conv_size * l3_conv_size * l2_num_convs, l3_conv_size * l3_conv_size * l3_num_convs, name='W_conv3')
#     b_conv3 = bias_variable([l3_num_convs])
#     h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#     h_pool3 = max_pool_2x2(h_conv3)
#
#     W_conv4 = weight_variable([l4_conv_size, l4_conv_size, l3_num_convs, l4_num_convs], l4_conv_size * l4_conv_size * l3_num_convs, l4_conv_size * l4_conv_size * l4_num_convs, name='W_conv4')
#     b_conv4 = bias_variable([l4_num_convs])
#     h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#     h_pool4 = max_pool_2x2(h_conv4)
#     conv_maxes = tf.reduce_max(h_pool4, [1, 2])
#     debug_layer = h_pool4
#
#     W_fc1 = weight_variable([widthD16 * heightD16 * l4_num_convs + 2 + numPulses, fc1_num_outs], widthD16 * heightD16 * l4_num_convs + 2 + numPulses, fc1_num_outs, name='W_fc1')
#     b_fc1 = bias_variable([fc1_num_outs])
#     h_pool5_flat = tf.reshape(h_pool4, [-1, widthD16 * heightD16 * l4_num_convs])
#     # h_pool5_odo_concat = tf.concat(1, [h_pool5_flat, odo*0.0, vel, pulse*config.use_odometer])
#     # Zero out odometer to experiment with odometer prediction.
#     h_pool5_odo_concat = tf.concat(1, [h_pool5_flat, odo*0.0, vel, pulse*0.0])
#
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool5_odo_concat, W_fc1) + b_fc1)
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#     num_outputs = 2 * max_log_outs
#     num_outputs += numPulses  # Add numpulses for odometer prediction.
#     W_fc4 = weight_variable([fc1_num_outs, num_outputs], fc1_num_outs, num_outputs, name='W_fc4')
#     b_fc4 = bias_variable([num_outputs])
#
#     output = (tf.matmul(h_fc1_drop, W_fc4) + b_fc4)
#     steering_softmax = tf.nn.softmax(output[:, :max_log_outs])
#     throttle_softmax = tf.nn.softmax(output[:, max_log_outs:max_log_outs + max_log_outs])  # for version without odometer prediction: output[:, max_log_outs:]
#     pulse_softmax = tf.nn.softmax(output[:, max_log_outs + max_log_outs:])  # odometer prediction
#
#     # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#     # http://stackoverflow.com/questions/33712178/tensorflow-nan-bug
#
#     steering_cross_entropy = -tf.reduce_mean(steering_ * tf.log(tf.clip_by_value(steering_softmax, 1e-10, 1.0)))
#     throttle_cross_entropy = -tf.reduce_mean(throttle_ * tf.log(tf.clip_by_value(throttle_softmax, 1e-10, 1.0)))
#     pulse_cross_entropy = -tf.reduce_mean(pulse * tf.log(tf.clip_by_value(pulse_softmax, 1e-10, 1.0)))
#     # cross_entropy = (steering_cross_entropy + throttle_cross_entropy) / 2.
#     # cross_entropy = ((steering_cross_entropy + throttle_cross_entropy) * train_mode + pulse_cross_entropy*0.6) / (0.6 + train_mode * 2.0)
#     cross_entropy = (steering_cross_entropy + throttle_cross_entropy*0.3 + pulse_cross_entropy*0.3) / 1.6
#     # L2 regularization
#     regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
#                     tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
#                     tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
#                     tf.nn.l2_loss(W_conv4) + tf.nn.l2_loss(b_conv4) +
#                     tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
#                     tf.nn.l2_loss(W_fc4) + tf.nn.l2_loss(b_fc4)
#                     )
#     # Add the regularization term to the loss.
#     cross_entropy += 0.00002 * regularizers # 5e-4
#     train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) # 1e-4
#     # train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy) # for 3-conv 0.000001
#     # train_step = tf.train.AdagradOptimizer(0.0002).minimize(cross_entropy) # for 3-conv 0.001
#     # train_step = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(cross_entropy)
#
#     # THIS NOW DOES A SUM OF SQUARES, NOT AN EXACT MATCH.
#     def compute_pred_accuracy(output, label):
#         prediction = tf.argmax(output, 1)
#         delta = (prediction - tf.argmax(label, 1))
#         energy = delta * delta
#         accuracy = tf.reduce_mean(tf.cast(energy, "float")) * 0.1  # arbitrary scale
#         # correctly_predicted = tf.equal(tf.argmax(label, 1), prediction)
#         # accuracy = tf.reduce_mean(tf.cast(correctly_predicted, "float"))
#
#         return prediction, accuracy
#
#     steering_pred, steering_accuracy = compute_pred_accuracy(steering_softmax, steering_)
#     throttle_pred, throttle_accuracy = compute_pred_accuracy(throttle_softmax, throttle_)
#
#     return x, odo, vel, pulse, steering_, throttle_, keep_prob, train_mode, train_step, steering_pred, steering_accuracy, throttle_pred, throttle_accuracy, steering_softmax, throttle_softmax, pulse_softmax, conv_maxes, debug_layer
