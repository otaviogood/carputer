import tensorflow as tf
import numpy as np

def weight_variable(shape, name=None):
    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def weight_variable(shape, fanIn, fanOut, name=None):
    return tf.Variable(tf.random_normal(shape) / np.sqrt(fanIn / 2),
                       name=name)#,
                       #collections='htmlize')


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



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
widthD32 = width / 32
heightD32 = height / 32
max_log_outs = 15
img_channels = 3
fc1_num_outs = 256 # 256
fc2_num_outs = 128 # 256
fc3_num_outs = 128 # 256
l1_conv_size = 3
l1_num_convs = 4   #16
l2_conv_size = 3
l2_num_convs = 8 #32
l3_conv_size = 3
l3_num_convs = 16   #64
l4_conv_size = 3
l4_num_convs = 32  #128
l5_conv_size = 3
l5_num_convs = 64  #128

numPulses = 128
pulseScale = 16.0

h_pool5_odo_concat = None


def gen_graph_ops():
    global h_pool5_odo_concat
    x = tf.placeholder(tf.float32, shape=[None, width * height * img_channels], name='x')
    odo = tf.placeholder(tf.float32, shape=[None, 1], name='odo')
    vel = tf.placeholder(tf.float32, shape=[None, 1], name='vel')
    pulse = tf.placeholder(tf.float32, shape=[None, numPulses], name='pulse')
    steering_ = tf.placeholder("float", shape=[None, max_log_outs], name='steering_')
    throttle_ = tf.placeholder("float", shape=[None, max_log_outs], name='throttle_')

    W_conv1 = weight_variable([l1_conv_size, l1_conv_size, img_channels, l1_num_convs], l1_conv_size * l1_conv_size * img_channels, l1_conv_size * l1_conv_size * l1_num_convs, name='W_conv1')
    b_conv1 = bias_variable([l1_num_convs])
    x_image = tf.reshape(x, [-1, width, height, img_channels])
    h_conv1 = tf.nn.relu(conv2d(x_image / 255.0 - 0.5, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)#, name='h_pool1', collections='htmlize')

    W_conv2 = weight_variable([l2_conv_size, l2_conv_size, l1_num_convs, l2_num_convs], l2_conv_size * l2_conv_size * l1_num_convs, l2_conv_size * l2_conv_size * l2_num_convs, name='W_conv2')
    b_conv2 = bias_variable([l2_num_convs])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([l3_conv_size, l3_conv_size, l2_num_convs, l3_num_convs], l3_conv_size * l3_conv_size * l2_num_convs, l3_conv_size * l3_conv_size * l3_num_convs, name='W_conv3')
    b_conv3 = bias_variable([l3_num_convs])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([l4_conv_size, l4_conv_size, l3_num_convs, l4_num_convs], l4_conv_size * l4_conv_size * l3_num_convs, l4_conv_size * l4_conv_size * l4_num_convs, name='W_conv4')
    b_conv4 = bias_variable([l4_num_convs])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    # W_conv5 = weight_variable([l5_conv_size, l5_conv_size, l4_num_convs, l5_num_convs], l5_conv_size * l5_conv_size * l4_num_convs, l5_conv_size * l5_conv_size * l5_num_convs, name='W_conv5')
    # b_conv5 = bias_variable([l5_num_convs])
    # h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    # h_pool5 = max_pool_2x2(h_conv5)

    keep_prob = tf.placeholder("float")

    W_fc1 = weight_variable([widthD16 * heightD16 * l4_num_convs + 2 + numPulses, fc1_num_outs], widthD16 * heightD16 * l4_num_convs + 2 + numPulses, fc1_num_outs, name='W_fc1')
    b_fc1 = bias_variable([fc1_num_outs])
    h_pool5_flat = tf.reshape(h_pool4, [-1, widthD16 * heightD16 * l4_num_convs])
    h_pool5_odo_concat = tf.concat(1, [h_pool5_flat, odo*0.0, vel, pulse])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_odo_concat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([fc1_num_outs, fc2_num_outs], fc1_num_outs, fc2_num_outs, name='W_fc2')
    b_fc2 = bias_variable([fc2_num_outs])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([fc2_num_outs, fc3_num_outs], fc2_num_outs, fc3_num_outs, name='W_fc3')
    b_fc3 = bias_variable([fc3_num_outs])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    num_outputs = 2 * max_log_outs
    W_fc4 = weight_variable([fc3_num_outs, num_outputs], fc3_num_outs, num_outputs, name='W_fc4')
    b_fc4 = bias_variable([num_outputs])

    output = tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
    steering = output[:, :max_log_outs]
    throttle = output[:, max_log_outs:]

    # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    # http://stackoverflow.com/questions/33712178/tensorflow-nan-bug

    steering_cross_entropy = -tf.reduce_sum(steering_ * tf.log(tf.clip_by_value(steering, 1e-10, 1.0)))
    throttle_cross_entropy = -tf.reduce_sum(throttle_ * tf.log(tf.clip_by_value(throttle, 1e-10, 1.0)))
    cross_entropy = (steering_cross_entropy + throttle_cross_entropy) / 2.
    # L2 regularization
    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                    tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
                    tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
                    tf.nn.l2_loss(W_conv4) + tf.nn.l2_loss(b_conv4) +
                    # tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(b_conv5) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2) +
                    tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3) +
                    tf.nn.l2_loss(W_fc4) + tf.nn.l2_loss(b_fc4)
                    )
    # Add the regularization term to the loss.
    cross_entropy += 0.1 * regularizers # 5e-4
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) # 1e-4
    # train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy) # for 3-conv 0.000001
    # train_step = tf.train.AdagradOptimizer(0.0002).minimize(cross_entropy) # for 3-conv 0.001
    # train_step = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(cross_entropy)

    def compute_pred_accuracy(output, label):
        prediction = tf.argmax(output, 1)
        correctly_predicted = tf.equal(tf.argmax(label, 1), prediction)
        accuracy = tf.reduce_mean(tf.cast(correctly_predicted, "float"))

        return prediction, accuracy

    steering_pred, steering_accuracy = compute_pred_accuracy(steering, steering_)
    throttle_pred, throttle_accuracy = compute_pred_accuracy(throttle, throttle_)

    return x, odo, vel, pulse, steering_, throttle_, keep_prob, train_step, steering_pred, steering_accuracy, throttle_pred, throttle_accuracy
