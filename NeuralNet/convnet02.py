"""Creates a model from training data.

Usage:
  convnet02.py [--indir=<path>] [--outdir=<path>]

Options:
  --indir=<path>   path to input npy files [default: ~/training-data]
  --outdir=<path>  [default: ~/convnet02-results]
"""

import os
import random
import time

from docopt import docopt
# http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import convnetshared1 as convshared
import html_output

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
import math

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

def generate_color_text(prev_val, cur_val):
    """
        Pretty formats and colorizes an input based on the delta from the previous.
        Used for debug statements

        return string
    """
    delta = cur_val - prev_val
    if(delta >= 0):
        return (bcolors.OKGREEN + format(cur_val, ".3f") + bcolors.ENDC)
    else:
        return (bcolors.FAIL + format(cur_val, ".3f") + bcolors.ENDC) 


# Parse args.
args = docopt(__doc__)
# Save data to an output dir.
outdir = os.path.expanduser(args['--outdir'])
output_path = os.path.join(outdir, time.strftime('%Y_%m_%d__%H_%M_%S_%p'))
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "layer_activations"))



# Load the filemashed data.
inpath = os.path.expanduser(args['--indir'])
bigArray         = np.load(inpath + "/picArray.npy")
gtArray          = np.load(inpath + "/gtArray.npy")
gtThrottlesArray = np.load(inpath + "/gtThrottlesArray.npy")
odoArray         = np.load(inpath + "/odoArray.npy")
velArray         = np.load(inpath + "/velArray.npy")
gtSoftArray = np.zeros((len(gtArray), convshared.max_log_outs), dtype=np.float32)
gtSoftThrottlesArray = np.zeros((len(gtThrottlesArray), convshared.max_log_outs), dtype=np.float32)
for i in xrange(len(gtArray)):
    gtVal = gtArray[i]
    gtSoftArray[i, gtVal] = 1.0
for i in xrange(len(gtThrottlesArray)):
    gtVal = int(gtThrottlesArray[i])
    gtSoftThrottlesArray[i, gtVal] = 1.0
pulseArray = np.zeros((len(gtArray), convshared.numPulses), dtype=np.float32)
if config.use_odometer != 0.0:
    print "setting up odometer. slow."
    for i in xrange(len(odoArray)):
        current_odo = odoArray[i]*1000.0 / convshared.pulseScale  # scale it so the whole track is in range.
        assert(current_odo < convshared.numPulses)
        for x in xrange(convshared.numPulses):
            # http://thetamath.com/app/y=max(0.0,1-abs((x-2)))
            pulseArray[i, x] = max(0.0, 1 - abs(current_odo - x))
    print "done odometer."

numTest = 4000
skipTest = 2
if config.running_on_laptop:
    numTest = 384 * 2
    skipTest = 8
testImages = bigArray[-numTest::skipTest]
testGT = gtSoftArray[-numTest::skipTest]
testGTThrottles = gtSoftThrottlesArray[-numTest::skipTest]
testOdo = odoArray[-numTest::skipTest][:, np.newaxis]
testVel = velArray[-numTest::skipTest][:, np.newaxis]
testPulses = pulseArray[-numTest::skipTest]
trainingImages = bigArray[0:-numTest]
trainingGT = gtSoftArray[0:-numTest]
trainingGTThrottles = gtSoftThrottlesArray[0:-numTest]
trainingOdo = odoArray[0:-numTest]
trainingVel = velArray[0:-numTest]
trainingPulses = pulseArray[0:-numTest]

# b164 = Image.frombuffer('RGB', (32, 32), bigArray[0].astype(np.int8), 'raw', 'RGB', 0, 1)
# b164.save("testtestT.png")

x, odo, vel, pulse, steering_, throttle_, keep_prob, train_step, steering_pred, steering_accuracy, throttle_pred, throttle_accuracy, steering_softmax, throttle_softmax, pooling_layers = convshared.gen_graph_ops()

timeStamp = time.strftime("%Y_%m_%d__%H_%M_%S")

sess = tf.Session()
sess.run(tf.initialize_all_variables())

def plotMultiFilter(units, layer_name):
    total_filters = []
    n_rows = 0
    for unit in units:
        sub_filters = unit.shape[3]
        total_filters.append(sub_filters)
        n_rows += sub_filters

    plt.figure(num=None, figsize=(16, 12), dpi=80)
    plt.title("Garbage Boy")
    n_columns = 6
    n_rows = math.ceil((n_rows + len(total_filters))/ n_columns) + 1
    
    plot_index = 1
    for i in range(0, len(total_filters)):
        if i > 0:
            # Split the layers visualized 
            plot_index = plot_index + n_columns
        for k in range(total_filters[i]):
            if plot_index > (n_rows * n_columns):
                plot_index = plot_index - 1
            plt.subplot(n_rows, n_columns, plot_index)
            plot_index = plot_index + 1
            plt.axis("off")
            plt.imshow(units[i][0,:,:,k], interpolation="nearest", cmap="gray")
    layer_file_name = layer_name + ".png"
    plt.savefig(os.path.join(output_path, "layer_activations", layer_file_name))

def get_activations(layer, stimuli):
    return sess.run(layer, feed_dict=stimuli)
    
def get_units(layers, layer_name, stimuli):
    units = []
    for l in range(0, len(layers)):
        layer = layers[l]
        unit = get_activations(layer, stimuli)
        units.append(unit)
    plotMultiFilter(units, layer_name)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# saver.restore(sess, "modelMed.ckpt")

all_xs = [image for image in testImages]
all_ys = [gt for gt in testGT]
all_ys_throttles = [gtt for gtt in testGTThrottles]
all_odos = testOdo
all_vels = testVel
all_pulses = testPulses

test_feed_dict = {
    x: all_xs,
    steering_: all_ys,
    throttle_: all_ys_throttles,
    keep_prob: 1.0,
    odo: all_odos,
    vel: all_vels,
    pulse: all_pulses,
}

allAccuracyTrain = []
allAccuracyTest = []

print '\n\n-----------------------'
print 'saving output data to %s' % output_path
print 'training on %s images' % len(trainingGT)
print '-----------------------\n'

random.seed(111)
iteration = 0
accuracy_check_iterations = []
sliding_window = []
sliding_window_size = 16
sliding_window_graph = []

#Headers for the console debug output
debug_header_list = ['Iteration', 'Test Accuracy', 'Test Throttle Accuracy', 'Sliding Average', 'Elapsed Time']
print '%s' % ' | '.join(map(str, debug_header_list))

#Vars to calculate deltas between iterations
prev_acc = 0
prev_throttle_acc = 0
prev_sliding_window = 0

start_time = time.time()
while iteration < 100000:
    randIndexes = random.sample(xrange(len(trainingGT)), min(64, len(trainingGT)))
    batch_xs = [trainingImages[index] for index in randIndexes]
    batch_ys = [trainingGT[index] for index in randIndexes]
    batch_ys_t = [trainingGTThrottles[index] for index in randIndexes]
    batch_odo = [trainingOdo[index] for index in randIndexes]
    batch_vel = [trainingVel[index] for index in randIndexes]
    batch_pulse = [trainingPulses[index] for index in randIndexes]
    bxs = np.array(batch_xs)
    bys = np.array(batch_ys)
    bys_t = np.array(batch_ys_t)
    bodos = np.array(batch_odo)[:, np.newaxis]
    bvels = np.array(batch_vel)[:, np.newaxis]
    bpulses = np.array(batch_pulse)
    # odos = np.zeros((bxs.shape[0], 1)).astype(np.float32)
    train_feed_dict = {x: bxs, steering_: bys, throttle_:bys_t, keep_prob: 0.5, odo: bodos, vel:bvels, pulse:bpulses}
    [_, train_acc] = sess.run([train_step, steering_accuracy], feed_dict=train_feed_dict)

    

    # Check the accuracy occasionally.
    if ((iteration % 256) == 255) or (iteration < 4):
        accuracy_check_iterations.append(iteration)
        allAccuracyTrain.append(train_acc)  # WARNING: this is running with dropout - should rerun if we want accuracy.

        [acc,
         throttle_acc,
         results_steering,
         results_throttle,
         results_steering_softmax,
         results_throttle_softmax] = sess.run([steering_accuracy,
                                               throttle_accuracy,
                                               steering_pred,
                                               throttle_pred,
                                               steering_softmax,
                                               throttle_softmax], feed_dict=test_feed_dict)
        allAccuracyTest.append(acc)
        sliding_window.append(acc)
        if len(sliding_window) > sliding_window_size: sliding_window = sliding_window[1:]
        sliding_window_graph.append(sum(sliding_window)/len(sliding_window))

        # print results
        # print results_throttle
        html_output.write_html(
            output_path, results_steering, results_throttle, all_xs, all_ys, all_ys_throttles,
            all_odos, convshared.width, convshared.height, tf.get_default_graph(),
            testImages, sess, test_feed_dict, results_steering_softmax, results_throttle_softmax)

        plt.clf()
        plt.plot(accuracy_check_iterations, allAccuracyTrain, 'bo')
        plt.plot(accuracy_check_iterations, allAccuracyTrain, 'b-')
        plt.plot(accuracy_check_iterations, allAccuracyTest, 'ro')
        plt.plot(accuracy_check_iterations, allAccuracyTest, 'r-')
        plt.plot(accuracy_check_iterations, sliding_window_graph, 'g-')
        # if len(allAccuracyTrain) < 100:
        #     plt.xlim([0, 100])
        # else:
        #     plt.xlim([0, len(allAccuracyTrain)])
        axes = plt.gca()
        axes.set_ylim([0, 1.05])
        plt.title("training (blue), test (red), avg " + str(round(sliding_window_graph[-1], 5)) + "  /  " + str(len(sliding_window)))
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(output_path, "progress.png"))

        # Save the model.
        save_path = saver.save(sess, os.path.join(output_path, "model.ckpt"))
        config.store('last_tf_model', save_path)

        # put the print after writing everything so it indicates things have been written.
        debug_iteration = format(iteration, '^10')

        #Format accuracy
        debug_acc = generate_color_text(prev_acc, acc)
        debug_acc = format(debug_acc, '^24')
        prev_acc = acc

        #Format throttle accuracy
        debug_throttle_acc = generate_color_text(prev_throttle_acc, throttle_acc)
        debug_throttle_acc = format(debug_throttle_acc, '^33')
        prev_throttle_acc = throttle_acc

        #Format sliding window 
        debug_sliding_window = generate_color_text(prev_sliding_window, sliding_window_graph[-1])
        debug_sliding_window = format(debug_sliding_window, '^25')
        prev_sliding_window = sliding_window_graph[-1]

        elapsed_time = format(format(time.time() - start_time, ".2f"), '^17')

        #Print everything
        print("%s %s %s %s %s" % (debug_iteration, debug_acc, debug_throttle_acc, debug_sliding_window, elapsed_time))
        start_time = time.time()

        # Visualize layer acivations 
        if config.visualize_layer_activations:
            # do_activations(pooling_layers[0], train_feed_dict)
            layer_name = "h_pool_" + str(iteration)
            get_units(pooling_layers, layer_name, train_feed_dict)
        

    # Increment.
    iteration += 1
