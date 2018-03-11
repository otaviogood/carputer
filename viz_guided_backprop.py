import sys,os
import matplotlib
import numpy as np
from matplotlib.pylab import *
matplotlib.use('Agg')
sys.path.append('..')
import tensorflow as tf
import config

from NeuralNet.convnetshared1 import NNModel
from NeuralNet.data_model import TrainingData

if __name__ == '__main__':
	net_model = NNModel()

	tf_config = tf.ConfigProto(device_count = {'GPU':config.should_use_gpu})
	sess = tf.Session(config=tf_config)

	# Add ops to save and restore all of the variables
	saver = tf.train.Saver()

	# Load the model checkpoint file
	try:
		tmp_file = config.tf_checkpoint_file
		print("Loading model from config: {}".format(tmp_file))
	except:
		tmp_file = config.load('last_tf_model') #gets the cached last tf trained model
		print "loading latest trained model: " + str(tmp_file)
		# print("CAN'T FIND THE model in config.")
		# sys.exit(-1)

	# Try to restore a session
	try:
		saver.restore(sess, tmp_file)
	except:
		print("Error restoring TF model: {}".format(tmp_file))
		sys.exit(-1)

	image_tensor = net_model.in_image
	G = tf.gradients(net_model.steering_regress_result, image_tensor)

	resized = np.zeros((128,128,3))
	batch = TrainingData.fromfilename('test',os.path.expanduser('~')+'/training-data/')

	for i in range(batch.NumSamples()):
		print("{} of {}".format(i, batch.NumSamples()))
		test_batch = batch.GenNoisyBatch([i])

		[flattened_gradient_image] = sess.run(
		    [G],  feed_dict=test_batch.FeedDict(net_model))[0]

		
		flattened_gradient_image = np.mean(flattened_gradient_image, 0)

		gradient_image = flattened_gradient_image.reshape((128,128,3))

		# WHAT IS THIS?
		# gradient_image_mono = np.sqrt(gradient_image[:,:,:]**2).sum(axis=2)
		gradient_image_mono = gradient_image[:,:,:].sum(axis=2)
		gradient_image_mono_neg = np.maximum(0.0, -gradient_image_mono)
		gradient_image_mono = np.maximum(0.0, gradient_image_mono)

		gradient_image_mono_norm = (gradient_image_mono-gradient_image_mono.min())/(gradient_image_mono.max()-gradient_image_mono.min())
		gradient_image_mono_norm_neg = (gradient_image_mono_neg-gradient_image_mono_neg.min())/(gradient_image_mono_neg.max()-gradient_image_mono_neg.min())

		clip_percent = 0.5
		# clip_min = gradient_image_mono_norm.max()*clip_percent# Top energy?
		gradient_image_mono_norm[gradient_image_mono_norm<clip_percent] = 0
		gradient_image_mono_norm_neg[gradient_image_mono_norm_neg<clip_percent] = 0

		scale = np.zeros(list(gradient_image_mono_norm.shape)+[3], dtype=np.float32)
		scale[:,:,0] = gradient_image_mono_norm_neg
		scale[:,:,1] = gradient_image_mono_norm
		scale[:,:,2] = 0.0

		image_rgb = batch.pic_array[i].reshape((128,128,3))
		image_rgb_f = image_rgb.astype(np.float32) * 0.5  # dim a little so you can see debugging colors

		#figure(figsize=(20,20))
		#subplot(1,2,1)
		#imshow(scale)
		#print scale.min(),scale.max()

		#subplot(1,2,2)
		#imshow(image_rgb)

		dbg = image_rgb_f + scale*255
		dbg[dbg > 255] = 255
		dbg = dbg.astype(np.uint8)

		fig = figure(figsize=(8,8))
		imshow(dbg)
		frame_number = str(i).zfill(5)
		savefig('./analysis/visuals/tmp_%s.png'%frame_number)
		print("Saving image: {}".format(frame_number))

	print("Done:")




