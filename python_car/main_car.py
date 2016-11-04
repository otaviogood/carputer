"""Records training data and / or drives the car with tensorflow.

Usage:
	main_car.py record
	main_car.py tf
"""

import math
import os
import re
import sys
import time

import cv2
from docopt import docopt
import numpy as np
import serial
import tensorflow as tf

import camera
import key_watcher
import NeuralNet.convnetshared1 as convshared


# Get args.
args = docopt(__doc__)


# Check the mode: recording vs TF driving vs TF driving + recording.
if args['record']:
	we_are_autonomous = False
	we_are_recording = True
	print("\n------ Ready to record training data ------\n")
elif args['tf']:
	we_are_autonomous = True
	we_are_recording = True
	print("\n****** READY TO DRIVE BY NEURAL NET and record data ******\n")


# Set up camera and key watcher.
camera_stream = camera.CameraStream().start()
last_key = ['']
key_watcher.KeyWatcher(last_key).start()


# Setup buffers and vars used by arduinos.
buffer_in = ''
buffer_out = ''
milliseconds = 0.0
last_odometer_reset = 0
odometer_ticks = 0
button_arduino_out = 0
button_arduino_in = 0


def setup_serial_and_reset_arduinos():
	# This will set up the serial ports. If they are already set up, it will
	# reset them, which also resets the Arduinos.
	print("Setting up serial and resetting Arduinos.")
	# On MacOS, you can find your Arduino via Terminal with
	# ls /dev/tty.*
	# then you can read that serial port using the screen command, like this
	# screen /dev/tty.[yourSerialPortName] [yourBaudRate]
	if os.name == 'nt':
		name_in = 'COM3'
		name_out = 'COM4'
	else:
		name_in = '/dev/tty.usbmodem14231'  # 5v Arduino Uno (16 bit)
		name_out = '/dev/tty.usbmodem14221'  # 3.3v Arduino Due (32 bit)
	# 5 volt Arduino Duemilanove, radio controller for input.
	port_in = serial.Serial(name_in, 38400, timeout=0.0)
	# 3 volt Arduino Due, servos for output.
	port_out = serial.Serial(name_out, 115200, timeout=0.0)
	# Flush for good luck. Not sure if this does anything. :)
	port_in.flush()
	port_out.flush()
	print("Serial setup complete.")
	return port_in, port_out


def make_data_folder(base_path):
	# Make a new dir to store data.
	base_path = os.path.expanduser(base_path)
	session_dir_name = time.strftime('%Y_%m_%d__%I_%M_%S_%p')
	session_full_path = os.path.join(base_path, session_dir_name)
	if not os.path.exists(session_full_path):
		os.makedirs(session_full_path)
	return session_full_path


def process_input(port_in, port_out):
	"""Reads steering, throttle, aux1 and button data reported from the arduinos.

	Returns: (steering, throttle, button_arduino_in, button_arduino_out)

	Return values may be None if the data from the arduino isn't related to the
	steering or throttle.
	"""
	# Input is buffered because sometimes partial lines are read
	global button_arduino_in, button_arduino_out, buffer_in, buffer_out, odometer_ticks, milliseconds
	try:
		buffer_in += port_in.read(port_in.in_waiting).decode('ascii')
		buffer_out += port_out.read(port_out.in_waiting).decode('ascii')
	except UnicodeDecodeError:
		# We can rarely get bad data over the serial port. The error looks like this:
		# buffer_in += port_in.read(port_in.in_waiting).decode('ascii')
		# UnicodeDecodeError: 'ascii' codec can't decode byte 0xf0 in position 0: ordinal not in range(128)
		buffer_in = ''
		buffer_out = ''
		print("Mysterious serial port error. Let's pretend it didn't happen. :)")
	# Init steering, throttle and aux1.
	steering, throttle, aux1 = None, None, None
	# Read lines from input Arduino
	while '\n' in buffer_in:
		line, buffer_in = buffer_in.split('\n', 1)
		match = re.search(r'(\d+) (\d+) (\d+)', line)
		if match:
			steering = int(match.group(1))
			throttle = int(match.group(2))
			aux1 = int(match.group(3))
		if line[0:1] == 'S':
			# This is just a toggle button
			button_arduino_in = 1 - button_arduino_in
			print "ButtonAIn toggle"
	# Read lines from output Arduino
	while '\n' in buffer_out:
		line, buffer_out = buffer_out.split('\n', 1)
		if line[0:3] == 'Mil':
			sp = line.split('\t')
			milliseconds = int(sp[1])
			odometer_ticks += 1
		if line[0:6] == 'Button':
			sp = line.split('\t')
			button_arduino_out = int(sp[1])
	return steering, throttle, aux1, button_arduino_in, button_arduino_out


def process_output(old_steering, old_throttle, steering, throttle, port_out):
	# Adjust the steering and throttle.
	throttle = 90 if 88 <= throttle <= 92 else min(throttle, 110)
	# Update steering
	if old_steering != steering:
		port_out.write(('S%d\n' % steering).encode('ascii'))
	# Update throttle
	if old_throttle != throttle:
		port_out.write(('D%d\n' % throttle).encode('ascii'))
	# Send keepalive.
	port_out.write(('keepalive\n').encode('ascii'))
	# Write all.
	port_out.flush()


def invert_log_bucket(a):
	# Reverse the function that buckets the steering for neural net output.
	# This is half in filemash.py and a bit in convnet02.py (maybe I should fix)
	# steers[-1] -= 90
	# log_steer = math.copysign( math.log(abs(steers[-1])+1, 2.0) , steers[-1]) # 0  -> 0, 1  -> 1, -1 -> -1, 2  -> 1.58, -2 -> -1.58, 3  -> 2
	# gtVal = gtArray[i] + 7
	steer = a - 7
	original = steer
	steer = abs(steer)
	steer = math.pow(2.0, steer)
	steer -= 1.0
	steer = math.copysign(steer, original)
	steer += 90.0
	steer = max(0, min(179, steer))
	return steer


# TOGGLE THIS TO RUN TENSORFLOW
if True:
	x, odo, vel, pulse, steering_, throttle_, keep_prob, train_step, steering_pred, steering_accuracy, throttle_pred, throttle_accuracy = convshared.gen_graph_ops()
	sess = tf.Session()
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	# tempfile = args['<path-to-model>'] #+ "/model.ckpt"
	# tempfile = "/Users/otaviogood/sfe-models/first-run-0937.ckpt"
	#tempfile = "/Users/otaviogood/sfe-models/golden/turning-bias-cleaned-data-0654.ckpt"
	# tempfile = "/Users/otaviogood/sfe-models/last-model-1213.ckpt"
	tempfile = "/Users/otaviogood/sfe-models/last-last-model-1251.ckpt"
	print "loading model: " + tempfile
	saver.restore(sess, tempfile)

	def do_tensor_flow(frame, odo_relative_to_start, speed):
		# Take a camera frame as input, send it to the neural net, and get steering back.
		resized = cv2.resize(frame, (128, 128))
		assert resized.shape == (128, 128, 3)  # Must be correct size and RGB, not RGBA.
		resized = resized.ravel()  # flatten the shape of the tensor.
		resized = resized[np.newaxis]  # make a batch of size 1.
		# scale values to match what's in filemash.
		# lidar_frame = lidar_frame.ravel()
		# lidar_frame = lidar_frame[np.newaxis]

		odo_arr = np.array([odo_relative_to_start / 1000.0])[np.newaxis]
		vel_arr = np.array([speed * 10.0])[np.newaxis]
		pulse_arr = np.zeros((vel_arr.shape[0], convshared.numPulses))  # HACK HACK!!!
		current_odo = odo_relative_to_start / convshared.pulseScale  # scale it so the whole track is in range. MUST MATCH CONV NET!!!
		for num in xrange(convshared.numPulses):
			# http://thetamath.com/app/y=max(0.0,1-abs((x-2)))
			pulse_arr[0, num] = max(0.0, 1 - abs(current_odo - num))
		steering_result, throttle_result = sess.run([steering_pred, throttle_pred], feed_dict={x: resized, keep_prob: 1.0, odo: odo_arr, vel: vel_arr, pulse: pulse_arr})  # run tensorflow
		steer = invert_log_bucket(steering_result[0])
		throt = invert_log_bucket(throttle_result[0])
		return (steer, throt)


if False:
	ops = model.network.graph_ops(model.network.params)
	x_input = ops[0]
	steering_output = ops[3]
	throttle_output = ops[4]
	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, "model/model.cpkt")

	def do_tensor_flow(frame, odo, speed):
		# Take a camera frame as input, send it to the neural net, and get steering back.
		resized = cv2.resize(frame, (128, 128))
		assert resized.shape == (128, 128, 3)  # Must be correct size and RGB, not RGBA.
		resized = resized[np.newaxis]  # make a batch of size 1.
		steering, throttle = sess.run(
			[steering_output, throttle_output], feed_dict={x_input: resized})  # run tensorflow
		steering = int(steering[0][0] + 90)
		throttle = int(throttle[0][0] + 90)
		return steering, throttle


def main():
	global last_odometer_reset
	# Init some vars..
	old_steering = 0
	old_throttle = 0
	old_aux1 = 0
	steering = 90
	throttle = 90
	aux1 = 0
	frame_count = 0
	session_full_path = ''
	last_switch = 0
	button_arduino_out = 0
	currently_running = False
	override_autonomous_control = False
	train_on_this_image = True
	vel = 0.0
	last_odo = 0
	last_millis = 0.0

	# Setup ports.
	port_in, port_out = setup_serial_and_reset_arduinos()

	# Start the clock.
	drive_start_time = time.time()
	print 'Awaiting switch flip..'

	while True:
		loop_start_time = time.time()

		# Switch was just flipped.
		if last_switch != button_arduino_out:
			last_switch = button_arduino_out
			# See if the car started up with the switch already flipped.
			if time.time() - drive_start_time < 1:
				print 'Error: start switch in the wrong position.'
				sys.exit()

			if button_arduino_out == 1:
				currently_running = True
				print '%s: Switch flipped.' % frame_count
				last_odometer_reset = odometer_ticks
				if we_are_recording and (not we_are_autonomous):
					session_full_path = make_data_folder('~/training-images')
					# lidar_full_path = make_data_folder('~/lidar-images')
					print 'STARTING TO RECORD.'
					print 'Folder: %s' % session_full_path
				elif we_are_recording and we_are_autonomous:
					session_full_path = make_data_folder('~/tf-driving-images')
					# lidar_full_path = make_data_folder('~/tf-lidar-images')
					print 'DRIVING AUTONOMOUSLY and STARTING TO RECORD'
					print 'Folder: %s' % session_full_path
				else:
					print("DRIVING AUTONOMOUSLY (not recording).")
			else:
				print("%s: Switch flipped. Recording stopped." % frame_count)
				currently_running = False

		# Read input data from arduinos.
		new_steering, new_throttle, new_aux1, button_arduino_in, button_arduino_out = (
			process_input(port_in, port_out))
		if new_steering != None:
			steering = new_steering
		if new_throttle != None:
			throttle = new_throttle
		if new_aux1 != None:
			aux1 = new_aux1

		# Check to see if we should stop the car via the RC during TF control.
		# But also provide a way to re-engage autonomous control after an override.
		if we_are_autonomous and currently_running:
			if (steering > 130 or steering < 50) and throttle > 130:
				if not override_autonomous_control:
					print '%s: Detected RC override: stopping.' % frame_count
					override_autonomous_control = True
	        if abs(aux1 - old_aux1) > 400 and override_autonomous_control:
			    old_aux1 = aux1
			    print '%s: Detected RC input: re-engaging autonomous control.' % frame_count
			    override_autonomous_control = False

		# Check to see if we should reset the odometer via aux1 during manual
		# driving. This is Button E on the RC transmitter.
		# The values will swing from ~1100 to ~1900.
		if abs(aux1 - old_aux1) > 400:
			old_aux1 = aux1
			print '%s: Resetting the odometer.' % frame_count
			last_odometer_reset = odometer_ticks

		# Overwrite steering with neural net output in autonomous mode.
		# This seems to take about 10ms.
		if we_are_autonomous and currently_running:
			# Calculate velocity from odometer. Gets weird when stopped.
			if odometer_ticks != last_odo and milliseconds > last_millis:
				vel = (float(odometer_ticks) - last_odo) / (milliseconds - last_millis)
				if last_millis == 0 and last_odo == 0:
					vel = 0
				if odometer_ticks - last_odo > 50 or last_odo >= odometer_ticks:
					vel = 0
				last_odo = odometer_ticks
				last_millis = milliseconds
			# Read a frame from the camera.
			frame = camera_stream.read()
			steering, throttle = do_tensor_flow(frame, odometer_ticks - last_odometer_reset, vel)

		if we_are_recording and currently_running:
			# TODO(matt): also record vel in filename for tf?
			# Read a frame from the camera.
			frame = camera_stream.read()
			# Save image with car data in filename.
			cv2.imwrite("%s/" % session_full_path +
				"frame_" + str(frame_count).zfill(5) +
				"_thr_" + str(throttle) +
				"_ste_" + str(steering) +
				"_mil_" + str(milliseconds) +
				"_odo_" + str(odometer_ticks - last_odometer_reset).zfill(5) +
				".png", frame)
		else:
			frame = camera_stream.read()
			cv2.imwrite('/tmp/test.png', frame)


		if override_autonomous_control:
			# Full brake and neutral steering.
			throttle, steering = 0, 90

		# Send output data to arduinos.
		process_output(old_steering, old_throttle, steering, throttle, port_out)
		old_steering = steering
		old_throttle = throttle

		# Attempt to go at 30 fps. In reality, we could go slower if something hiccups.
		seconds = time.time() - loop_start_time
		while seconds < 1 / 30.:
			time.sleep(0.001)
			seconds = time.time() - loop_start_time
		frame_count += 1


if __name__ == '__main__':
	main()
