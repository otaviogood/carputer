"""Creates a model from training data.

Usage:
  convnet02.py [--indir=<path>] [--outdir=<path>]

Options:
  --indir=<path>   path to input npy files [default: ~/training-data]
  --outdir=<path>  [default: ~/convnet02-results]
"""

import os
import os.path
import random
import time
from shutil import copyfile

from docopt import docopt
# http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from convnetshared1 import NNModel
from convnetshared1 import LSTMModel
from data_model import TrainingData
from html_output import HtmlDebug

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

#Colorfied print statements
#USAGE: print bcolors.WARNING + "TEXT" + bcolors.ENDC
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def Colorize(prev_val, cur_val, format_str=None):
    """
        Pretty formats and colorizes an input based on the delta from the previous.
        Used for debug statements

        return string
    """
    delta = cur_val - prev_val
    if(delta >= 0):
        s = (bcolors.OKGREEN + format(cur_val, ".3f") + bcolors.ENDC)
    else:
        s = (bcolors.FAIL + format(cur_val, ".3f") + bcolors.ENDC)
    if format_str is None:
        return s
    else:
        return format(s, format_str)


# Parse args.
args = docopt(__doc__)
# Save data to an output dir.
outdir = os.path.expanduser(args['--outdir'])
output_path = os.path.join(outdir, time.strftime('%Y_%m_%d__%H_%M_%S_%p'))
if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Tensorflow version: " + tf.__version__)

# -------- Load all data --------
train_data = TrainingData.fromfilename("train", args['--indir'])
test_data = TrainingData.fromfilename("test", args['--indir'])
print("In dir: {}".format(args["--indir"]))

if config.neural_net_mode == 'alexnet':
    net_model = NNModel()
elif config.neural_net_mode == 'lstm':
    net_model = LSTMModel()
else: assert False  # Bad training mode in config.py.

numTest = 8000
skipTest = 1
if config.running_on_laptop:
    numTest = 8500# 384 * 8
    skipTest = 1
test_data.TrimArray(numTest, skipTest)
assert test_data.NumSamples() >= net_model.n_steps

timeStamp = time.strftime("%Y_%m_%d__%H_%M_%S")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# saver.restore(sess, "modelMed.ckpt")

allAccuracyTrain = []
allAccuracyTest = []

print '\n\n-----------------------'
print 'saving output data to %s' % output_path
print 'training on %s images' % train_data.NumSamples()
print 'testing on %s images' % test_data.NumSamples()
print '-----------------------\n'

random.seed(111)
iteration = 0
accuracy_check_iterations = []
sliding_window = []
sliding_window_size = 16
sliding_window_graph = []

#Headers for the console debug output
debug_header_list = ['Iteration', 'Elapsed Time', 'Sliding Average', 'steer regression', 'throttle regression']
print '%s' % ' | '.join(map(str, debug_header_list))

#Vars to calculate deltas between iterations
prev_sliding_window = 0
prev_squared_diff = 0.0
prev_squared_diff_throttle = 0.0
best_total_score = 1000000000.0

merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train', sess.graph)

start_time = time.time()
while iteration < 1000*128:
    if config.neural_net_mode == 'alexnet':
        randIndexes = random.sample(xrange(train_data.NumSamples()), min(TrainingData.batch_size, train_data.NumSamples()))
        batch = train_data.GenBatch(randIndexes)
        train_feed_dict = batch.FeedDict(net_model, 0.6)
        [summary_str, _, train_steer_diff] = sess.run(
            [merged_summaries, net_model.train_step, net_model.squared_diff], feed_dict=train_feed_dict)
        train_writer.add_summary(summary_str, iteration)

        # Check the accuracy occasionally.
        if ((iteration % 128) == 127):
            # print("train_steer_diff: " + str(train_steer_diff) + "    train_loss: " + str(train_loss))
            full_list = range(test_data.NumSamples() - (net_model.n_steps - 1))
            # Split full list into multiple batch_size lists plus the remainder list.
            test_list = [full_list[i:i + TrainingData.batch_size] for i in xrange(0, len(full_list), TrainingData.batch_size)]
            results_steering_regress = []
            results_throttle_regress = []
            results_squared_diff = 0.0
            results_squared_diff_throttle = 0.0
            results_loss = 0.0
            results_regularizers = 0.0
            for indexes in test_list:
                test_batch = test_data.GenBatch(indexes)
                test_feed_dict = test_batch.FeedDict(net_model, 1.0)

                [a0, b0, c0, d0, e0, f0] = sess.run([net_model.steering_regress_result,
                               net_model.throttle_regress_result,
                               net_model.squared_diff,
                               net_model.squared_diff_throttle,
                               net_model.loss,
                               net_model.regularizers,
                               ], feed_dict=test_feed_dict)
                results_steering_regress.extend(a0)
                results_throttle_regress.extend(b0)
                size = len(test_list)
                results_squared_diff += c0 / size
                results_squared_diff_throttle += d0 / size
                results_loss += e0 / size
                results_regularizers += f0 / size

            allAccuracyTest.append(results_squared_diff)
            sliding_window.append(results_squared_diff)
            if len(sliding_window) > sliding_window_size: sliding_window = sliding_window[1:]
            sliding_window_graph.append(sum(sliding_window) / len(sliding_window))

            html = HtmlDebug()
            html.write_line("Iteration: " + str(iteration) + "&nbsp;&nbsp; MSE steer: " + str(results_squared_diff) + "&nbsp;&nbsp; MSE throttle: " + str(results_squared_diff_throttle))

            plt_len = 2000
            plt_start = 50
            html.draw_graph([results_throttle_regress[plt_start:plt_len], test_data.throttle_array[plt_start+(net_model.n_steps-1):plt_len+(net_model.n_steps-1)]], 'THROTTLE   red: LSTM, green: labels    Iter: ' + str(iteration))
            html.draw_graph([results_steering_regress[plt_start:plt_len], test_data.steer_array[plt_start+(net_model.n_steps-1):plt_len+(net_model.n_steps-1)]], 'STEERING   red: LSTM, green: labels    Iter: ' + str(iteration))
            html.draw_graph([test_data.vel_array[plt_start+(net_model.n_steps-1):plt_len+(net_model.n_steps-1)]], 'Speed')

            html.write_html(test_data, tf.get_default_graph(),
                            sess, results_steering_regress, results_throttle_regress, net_model, test_data.FeedDict(net_model))
            html.write_file(output_path)

            accuracy_check_iterations.append(iteration)
            allAccuracyTrain.append(train_steer_diff)
            plt.plot(accuracy_check_iterations, allAccuracyTrain, 'bo')
            plt.plot(accuracy_check_iterations, allAccuracyTrain, 'b-')
            plt.plot(accuracy_check_iterations, allAccuracyTest, 'ro')
            plt.plot(accuracy_check_iterations, allAccuracyTest, 'r-')
            plt.plot(accuracy_check_iterations, sliding_window_graph, 'g-')
            axes = plt.gca()
            axes.set_ylim([0, 1000.05])
            plt.title("training (blue), test (red), avg " + str(round(sliding_window_graph[-1], 5)) + "  /  " + str(
                len(sliding_window)))
            plt.xlabel('iteration')
            plt.ylabel('diff squared')
            plt.savefig(os.path.join(output_path, "progress.png"))

            # Save the model.
            cool_score = results_squared_diff + results_squared_diff_throttle * 2.0
            if cool_score < best_total_score:
              best_total_score = cool_score
              save_path = saver.save(sess, os.path.join(output_path, "model.ckpt"))
              config.store('last_tf_model', save_path)
              copyfile(os.path.join(output_path, "debug.html"), os.path.join(output_path, "debug_best.html"))
              print("Saved: " + str(cool_score))

            # put the print after writing everything so it indicates things have been written.
            debug_iteration = format(iteration, '^10')
            debug_loss = format(results_loss, '^16')
            debug_regularizers = format(results_regularizers, '^16')

            # Format sliding window
            debug_sliding_window = Colorize(prev_sliding_window, sliding_window_graph[-1], '^24')
            prev_sliding_window = sliding_window_graph[-1]

            # Format steering regression accuracy
            debug_sqr_acc = Colorize(prev_squared_diff, results_squared_diff, '^28')
            prev_squared_diff = results_squared_diff
            # Format throttle regression accuracy
            debug_sqr_acc_throttle = Colorize(prev_squared_diff_throttle, results_squared_diff_throttle, '^30')
            prev_squared_diff_throttle = results_squared_diff_throttle

            elapsed_time = format(format(time.time() - start_time, ".2f"), '^16')

            # Print everything
            print("%s %s %s %s %s %s %s" % (
            debug_iteration, elapsed_time, debug_sliding_window, debug_sqr_acc, debug_sqr_acc_throttle, debug_loss,
            debug_regularizers))
            start_time = time.time()

    elif config.neural_net_mode == 'lstm':
        # elapsed_time = format(format(time.time() - start_time, ".2f"), '^16')
        # start_time = time.time()
        # print("SGD   " + elapsed_time)
        # Make input tensors - one hots
        randIndexes = random.sample(xrange(train_data.NumSamples() - net_model.n_steps), TrainingData.batch_size)
        batch = train_data.GenBatchLSTM(net_model, randIndexes)
        train_feed_dict = batch.FeedDict(net_model, 0.6)

        # elapsed_time = format(format(time.time() - start_time, ".2f"), '^16')
        # start_time = time.time()
        # print("batch " + elapsed_time)
        # Run optimization op (backprop)
        [_, train_steer_diff, train_loss] = sess.run([net_model.train_step, net_model.squared_diff, net_model.loss], feed_dict=train_feed_dict)

        if ((iteration % 128) == 127):
            print("TESTING")
            # print("train_steer_diff: " + str(train_steer_diff) + "    train_loss: " + str(train_loss))
            full_list = range(test_data.NumSamples() - (net_model.n_steps - 1))
            # Split full list into multiple batch_size lists plus the remainder list.
            test_list = [full_list[i:i + TrainingData.batch_size] for i in xrange(0, len(full_list), TrainingData.batch_size)]
            results_steering_regress = []
            results_throttle_regress = []
            results_squared_diff = 0.0
            results_squared_diff_throttle = 0.0
            results_loss = 0.0
            results_regularizers = []
            # for index in xrange(0, numTest - net_model.n_steps, net_model.n_steps):
            for indexes in test_list:
                test_batch = test_data.GenBatchLSTM(net_model, indexes)
                test_feed_dict = test_batch.FeedDict(net_model, 1.0)

                [a0, b0, c0, d0, e0, f0, x0, y0] = sess.run([net_model.steering_regress_result,
                               net_model.throttle_regress_result,
                               net_model.squared_diff,
                               net_model.squared_diff_throttle,
                               net_model.loss,
                               net_model.regularizers,
                               net_model.mid_act,
                               net_model.mid_lstm,
                               ], feed_dict=test_feed_dict)
                results_steering_regress.extend(a0)
                results_throttle_regress.extend(b0)
                size = len(test_list)
                results_squared_diff += c0 / size
                results_squared_diff_throttle += d0 / size
                results_loss += e0 / size
                results_regularizers += f0 / size

            allAccuracyTest.append(results_squared_diff)
            sliding_window.append(results_squared_diff)
            if len(sliding_window) > sliding_window_size: sliding_window = sliding_window[1:]
            sliding_window_graph.append(sum(sliding_window) / len(sliding_window))

            print("act  std: " + str(np.std(x0)) + " act  mean: " + str(np.mean(x0)))
            print("lstm std: " + str(np.std(y0)) + " lstm mean: " + str(np.mean(y0)))
            html = HtmlDebug()
            html.write_line("Iteration: " + str(iteration) + "&nbsp;&nbsp; MSE steer: " + str(results_squared_diff) + "&nbsp;&nbsp; MSE throttle: " + str(results_squared_diff_throttle))

            plt_len = 2000
            plt_start = 50
            html.draw_graph([results_throttle_regress[plt_start:plt_len], test_data.throttle_array[plt_start+(net_model.n_steps-1):plt_len+(net_model.n_steps-1)]], 'THROTTLE   red: LSTM, green: labels    Iter: ' + str(iteration))
            html.draw_graph([results_steering_regress[plt_start:plt_len], test_data.steer_array[plt_start+(net_model.n_steps-1):plt_len+(net_model.n_steps-1)]], 'STEERING   red: LSTM, green: labels    Iter: ' + str(iteration))
            html.draw_graph([test_data.vel_array[plt_start+(net_model.n_steps-1):plt_len+(net_model.n_steps-1)]], 'Speed')

            html.write_html(test_data, tf.get_default_graph(),
                            sess, results_steering_regress, results_throttle_regress, net_model, test_feed_dict)
            html.write_file(output_path)
            # print("test: " + str(results_squared_diff) + "    " + str(results_squared_diff_throttle))

            accuracy_check_iterations.append(iteration)
            allAccuracyTrain.append(train_steer_diff)
            # plt.plot(accuracy_check_iterations, allAccuracyTrain, 'bo')
            # plt.plot(accuracy_check_iterations, allAccuracyTrain, 'b-')
            # plt.plot(accuracy_check_iterations, allAccuracyTest, 'ro')
            # plt.plot(accuracy_check_iterations, allAccuracyTest, 'r-')
            # plt.plot(accuracy_check_iterations, sliding_window_graph, 'g-')
            # axes = plt.gca()
            # axes.set_ylim([0, 1000.05])
            # plt.title("training (blue), test (red), avg " + str(round(sliding_window_graph[-1], 5)) + "  /  " + str(
            #     len(sliding_window)))
            # plt.xlabel('iteration')
            # plt.ylabel('diff squared')
            # plt.savefig(os.path.join(output_path, "progress.png"))

            # Save the model.
            cool_score = results_squared_diff + results_squared_diff_throttle * 2.0
            if cool_score < best_total_score:
              best_total_score = cool_score
              save_path = saver.save(sess, os.path.join(output_path, "model.ckpt"))
              config.store('last_tf_model', save_path)
              copyfile(os.path.join(output_path, "debug.html"), os.path.join(output_path, "debug_best.html"))
              print("Saved: " + str(cool_score))

            # put the print after writing everything so it indicates things have been written.
            debug_iteration = format(iteration, '^10')
            debug_loss = format(results_loss, '^16')
            debug_regularizers = format(results_regularizers, '^16')

            # Format sliding window
            debug_sliding_window = Colorize(prev_sliding_window, sliding_window_graph[-1], '^24')
            prev_sliding_window = sliding_window_graph[-1]

            # Format steering regression accuracy
            debug_sqr_acc = Colorize(prev_squared_diff, results_squared_diff, '^28')
            prev_squared_diff = results_squared_diff
            # Format throttle regression accuracy
            debug_sqr_acc_throttle = Colorize(prev_squared_diff_throttle, results_squared_diff_throttle, '^30')
            prev_squared_diff_throttle = results_squared_diff_throttle

            elapsed_time = format(format(time.time() - start_time, ".2f"), '^16')

            # Print everything
            print("%s %s %s %s %s %s %s" % (
            debug_iteration, elapsed_time, debug_sliding_window, debug_sqr_acc, debug_sqr_acc_throttle, debug_loss,
            debug_regularizers))
            start_time = time.time()

    # Increment.
    iteration += 1
