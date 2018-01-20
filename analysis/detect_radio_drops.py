import cv2
from os import listdir
from os import rename
from os.path import isfile, join
import sys
import matplotlib.pyplot as plt
import numpy as np

# Get the data
path_to_images = sys.argv[1]
images = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]

# Data buckets
frames = []

# The last good stuff
# [frame, throttle, steering]
last_values = [0, 0, 0]
good_throttles = []
good_steerings = []

# The bad stuff
steerings = []
throttles = []
bad_frames = []

def save_the_day(image_name, path_to_image, new_throttle, new_steering):

	# full path to rename
	full_orig_path = join(path_to_image, image_name)

	# Create a valid image name
	split_data = image_name.split("_")
	split_data[3] = new_throttle
	split_data[5] = new_steering

	new_frame = "_".join(split_data)

	print("Changed {} to {}".format(image_name, new_frame))

	full_new_path = join(path_to_image, new_frame)
	print("Renaming {} to {}".format(full_orig_path, full_new_path))

	# Rename
	rename(full_orig_path, full_new_path)
	pass

# Iterate through all of the images, checking for misbehaving transmitter
# 0 Throttle is bad
# 180 Steering is bad
for image in images:

	full_path = join(path_to_images, image)

	# Split
	split_data = image.split("_")

	# Frame count
	frame = split_data[1]
	frames.append(frame)

	is_bad_frame = False

	# Throttle
	throttle = split_data[3]
	throttles.append(throttle)
	good_throttle = throttle
	if int(good_throttle) is 0:
		good_throttle = last_values[1]
		bad_frames.append(frame)
		is_bad_frame = True

	# Steering
	steering = split_data[5]
	steerings.append(steering)
	good_steering = steering
	if int(good_steering) is 180:
		good_steering = last_values[2]
		bad_frames.append(frame) 
		is_bad_frame = True

	# Populate for graphs
	good_throttles.append(good_throttle)
	good_steerings.append(good_steering)
	last_values = [frame, good_steering, good_throttle]

	# Fix the problem
	if is_bad_frame:
		print("Fixing a bad frame: {}".format(full_path))
		print("Throttle: {} to {} Steering: {} to {}".format(split_data[3], good_throttle, split_data[5], good_steering))
		save_the_day(image, path_to_images, good_throttle, good_steering)

print("Got {} bad data frames".format(len(bad_frames)))

plt.figure(1)

# Throttles
bad_throttle_plot = plt.subplot(221)
bad_throttle_plot.set_title("Bad throttles")
plt.plot(frames, throttles, color="r")

good_throttle_plot = plt.subplot(223)
good_throttle_plot.set_title("Good throttles")
plt.plot(frames, good_throttles, color="g")

# Steering
bad_steering_plot = plt.subplot(222)
bad_steering_plot.set_title("Bad steering")
plt.plot(frames, steerings, color="r")

good_steering_plot = plt.subplot(224)
good_steering_plot.set_title("Good steering")
plt.plot(frames, good_steerings, color="g")
plt.show()

