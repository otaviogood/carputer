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

# Parse args.
args = docopt(__doc__)
# Save data to an output dir.
outdir = os.path.expanduser(args['--outdir'])
output_path = os.path.join(outdir, time.strftime('%Y_%m_%d__%I_%M_%S_%p'))
if not os.path.exists(output_path):
    os.makedirs(output_path)


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
    gtVal = gtThrottlesArray[i]
    gtSoftThrottlesArray[i, gtVal] = 1.0
pulseArray = np.zeros((len(gtArray), convshared.numPulses), dtype=np.float32)
print "setting up odometer. slow."
for i in xrange(len(odoArray)):
    # pulses = np.zeros((numPulses), dtype=np.float64)
    current_odo = odoArray[i]*1000.0 / convshared.pulseScale  # scale it so the whole track is in range.
    assert(current_odo < convshared.numPulses)
    for x in xrange(convshared.numPulses):
        # http://thetamath.com/app/y=max(0.0,1-abs((x-2)))
        pulseArray[i, x] = max(0.0, 1 - abs(current_odo - x))
        # pulseArray[i, 0] = 1.0 * (i % 2)
print "done odometer."

numTest = 384*2
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

x, odo, vel, pulse, steering_, throttle_, keep_prob, train_step, steering_pred, steering_accuracy, throttle_pred, throttle_accuracy = convshared.gen_graph_ops()

timeStamp = time.strftime("%Y_%m_%d__%H_%M_%S")

sess = tf.Session()
sess.run(tf.initialize_all_variables())

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
    sess.run(train_step, feed_dict=train_feed_dict)

    # Check the accuracy occasionally.
    if ((iteration % 64) == 63) or (iteration < 4):
        accuracy_check_iterations.append(iteration)
        allAccuracyTrain.append(sess.run(steering_accuracy, feed_dict=train_feed_dict))

        # odosTest = np.zeros((len(all_xs), 1)).astype(np.float32)
        acc = sess.run(steering_accuracy, feed_dict=test_feed_dict)
        allAccuracyTest.append(acc)
        throttle_acc = sess.run(throttle_accuracy, feed_dict=test_feed_dict)

        results = sess.run(steering_pred, feed_dict=test_feed_dict)
        results_throttle = sess.run(throttle_pred, feed_dict=test_feed_dict)
        print results
        print results_throttle
        html_output.write_html(
            output_path, results, results_throttle, all_xs, all_ys, all_ys_throttles,
            all_odos, convshared.width, convshared.height, tf.get_default_graph(),
            testImages, sess, test_feed_dict)

        plt.plot(accuracy_check_iterations, allAccuracyTrain, 'bo')
        plt.plot(accuracy_check_iterations, allAccuracyTrain, 'b-')
        plt.plot(accuracy_check_iterations, allAccuracyTest, 'ro')
        plt.plot(accuracy_check_iterations, allAccuracyTest, 'r-')
        # if len(allAccuracyTrain) < 100:
        #     plt.xlim([0, 100])
        # else:
        #     plt.xlim([0, len(allAccuracyTrain)])
        axes = plt.gca()
        axes.set_ylim([0, 1.05])
        plt.title("training (blue), test (red)")
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(output_path, "progress.png"))

        # Save the model.
        save_path = saver.save(sess, os.path.join(output_path, "model.ckpt"))
        # put the print after writing everything so it indicates things have been written.
        print 'iteration, test accuracy, test throttle accuracy: %5.0f   %3.3f   %3.3f' % (iteration, acc, throttle_acc)

    # Increment.
    iteration += 1
