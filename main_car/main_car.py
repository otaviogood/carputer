#!/usr/bin/env python

"""
Main script for driving the kartputer autnomously and manually

Usage:
    main_car.py [--record]

"""
# System modules
import os
import platform
import signal
import subprocess
import sys
import time

# Third party modules
import argparse
import cv2
import serial

# Our modules
import camera
import debug_message as dm
dm.verbose = True

# Cartputer modules 
main_car_directory = os.path.dirname(os.path.realpath(__file__))
carputer_directory = os.path.dirname(main_car_directory)
nn_directory = os.path.join(carputer_directory, "NeuralNet")

# Configuration file
sys.path.append(carputer_directory)
import config

# Neural Network modules
# sys.path.append(nn_directory)
# from convnetshared1 import NNModel
# from data_model import TrainingData

#########################
# Constants and Globals #
#########################
# TODO (GM): Clean this up yo
is_running = True # main while loop boolean



################
# Utility Belt #
################
def clamp(value, min, max):
    if(value < min):
        return min
    if(value > max):
        return max
    return value

###################
# Setup functions #
###################
def check_for_insomnia():
    if platform.system() == "Darwin":
        dm.print_info("Checking for Insomnia (necessary for everything to work during lid close)")
        proc = subprocess.Popen(["ps aux"], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        if not "Insomnia" in out:
            dm.print_fatal("ERROR: YOU ARE NOT RUNNING InsomniaX.")
            dm.print_fatal("THAT IS THE PROGRAM THAT LETS YOU SHUT THE LID ON THE MAC AND KEEP IT RUNNING.")
            dm.print_fatal("How are you gonna drive a car if your driver is asleep?")
            sys.exit(0)

def parse_command_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", dest="record", action="store_true")
    args = parser.parse_args()

    return args.record

def setup_camera():
    """Initializes threaded camera class to read frames from the video stream

    """
    return camera.CameraStream(src=config.camera_id)

def setup_serial_port(port_name, baudrate):
    
    dm.print_info("Opening port: {} with baudrate: {}".format(port_name, baudrate))

    # Setup the port
    try:
        port = serial.Serial(port_name, baudrate, timeout=None)
    except(OSError, serial.SerialException):
        dm.print_fatal("Could not open serial port: {} with baud: {}".format(port_name, baudrate))
        sys.exit(-1)

    # Flush for good luck
    port.flush()
    return port

def signal_handler(*args):
    dm.print_warning("Ctrl-c detected, closing...")
    global is_running
    is_running = False
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

#############################
# Data Processing functions #
#############################
session_dir_name = None

def make_data_folder(base_path):
    """ Make directories for logging data
    
        ~/training_images = For data taken during teleop mode
        ~/tf_driving_images = For data taken during autonomous mode
    """
    global session_dir_name
    base_path = os.path.expanduser(base_path)
    session_dir_name = time.strftime('%Y_%m_%d__%I_%M_%S_%p')
    session_full_path = os.path.join(base_path, session_dir_name)
    if not os.path.exists(session_full_path):
        os.makedirs(session_full_path)
    
    return session_full_path    def clamp(value, min, max):
    if(value < min):
        return min
    if(value > max):
        return max
    return value

def init_data_logging(manual_dir, auto_dir):
    """Creates data logging directories and setups a dict for easy logging

    """
    manual_full_path = make_data_folder(manual_dir)
    auto_full_path = make_data_folder(auto_dir)

    logging_dict = {
        "manual": manual_full_path,
        "auto": auto_full_path
    }

    return logging_dict


def log_data(steering, throttle, logging_dict, logging_type):
    """ Log data to the logging dir specified
    """
    # Get the data to log
    global start_time

    session_full_path = logging_dict["dir_map"][logging_type]
    frame_count = logging_dict["frame_count"]
    milliseconds = logging_dict["milliseconds"]
    frame = logging_dict["frame"]
    odo_delta = logging_dict["odo_delta"]

    # Log it
    cv2.imwrite("%s/" % session_full_path +
				"frame_" + str(frame_count).zfill(5) +
				"_thr_" + str(throttle) +
				"_ste_" + str(steering) +
				"_mil_" + str(milliseconds) +
				"_odo_" + str(odo_delta).zfill(5) +
				".png", frame )


##########################
# Tensorflow Functions   #
##########################

##########################
# Main driving functions #
##########################
def send_vehicle_commands(steering, throttle, port):
    """
        Sends steering and throttle to the kart 
    """

    # Clamp steering
    steering = clamp(steering, STEERING_MIN_VALUE, STEERING_MAX_VALUE)

    # Clamp throttle
    throttle = clamp(throttle, THORTTLE_MIN_VALUE, THORTTLE_MAX_VALUE)

    # Encode data for Arduino
    steering_out = ('S%d\n' % steering).encode('ascii')
    throttle_out = ('T%d\n' % throttle).encode('ascii')

    # Write
    port.write(steering_out)
    port.flush()
    
    port.write(throttle_out)
    port.flush()

def stop_vehicle(port):
    """Sends a zero throttle and middle steering

        TODO: Perhaps keep the last steering value so we don't flip the vehicle? Would that happen at high speeds?
    """
    send_vehicle_commands(90, 0, port)

def drive_autonomously(session, net_model, car_port, logging_dict):
    """Main autonomous driving function. recieves results from tensorflow and dispatchs it to the car_port
    """
    # # Get our steering and throttle values from tensorflow
    # steering, throttle = do_tensorflow(session, net_model, logging_dict)

    # Send those steering and throttle values over serial
    send_vehicle_commands(steering, throttle, car_port)

    # log data
    log_data(steering, throttle, logging_dict, "auto")

def drive_manually(steering, throttle, port, logging_dict):
    # Arduino is expecting
    # steering,throttle

    # Map steering
    #mapped_steering = int(map_to(steering, -1.0, 1.0, STEERING_MIN_VALUE, STEERING_MAX_VALUE))
    # exp_steering = exponential_map_to(steering, 1.7)
    # mapped_steering = int(map_to(exp_steering, -1.0, 1.0, STEERING_MIN_VALUE, STEERING_MAX_VALUE))

    
    # # Artifical "deadband" for Xbox 360 controller
    # if mapped_steering <= 95 and mapped_steering >= 90:
    #     mapped_steering = 90

    # # Map throttle
    # # Deadband throttle
    # if throttle < -0.75:
    #     throttle = -0.75
    # mapped_throttle = int(map_to(throttle, -0.75, 1, THORTTLE_MIN_VALUE, THORTTLE_MAX_VALUE))

    # Send these values over the serial port
    send_vehicle_commands(mapped_steering, mapped_throttle, port)

    # Log the values
    log_data(mapped_steering, mapped_throttle, logging_dict, "manual")



def main():
    
    dm.print_info("Starting carputer...")

    dm.print_debug("Parsing command line arguments")
    # Parse command line looking for the record flag
    # TODO (GM): Add more command line options, camera id, serial port, etc
    # TODO (ALL): We should always be recording data, this flag decides where to dump the data
    record = parse_command_line()


    # Setup Serial ports

    # Setup camera
    camera_stream = setup_camera()
    camera_stream.start()
    frame = camera_stream.read()

    # Setup Tensorflow

    # Init the frame counter
    frame_count = 0

    # Init the time
    milliseconds = time.time() * 1000.0

    # init the delta in odometer
    odo_delta = 0

        # Setup data logging
    # Use this dict to share the common data between auto and manual driving
    logging_dir_map = init_data_logging(config.manual_driving_log_dir, config.auto_driving_log_dir)
    logging_dict = {
        "dir_map": logging_dir_map,
        "milliseconds": milliseconds,
        "frame_count": frame_count,
        "frame": frame,
        "odo_delta": odo_delta
    }

    if record:
        dm.print_info("Saving training images to: {}".format(logging_dir_map["manual"]))
    else:
        dm.print_info("This is an autonomous run. Saving images to {}".format(logging_dir_map["auto"]))
    

    # Main loop booleans
    global is_running
    is_autonomous = False # Is tensorflow driving
    killswitch_engaged = False # Do we need to stop NOW
    engage_autonomous_driving = False # Should we drop into tensorflow driving?
    engage_killswitch = False # Should we stop NOW

    while is_running:
        # Start the loop timer
        loop_start_time = time.time()
        dm.print_debug("In main loop")

        # Grab a frame from the camera
        # Default size from read() is [320, 240]
        logging_dict["frame"] = camera_stream.read()

        # Read values from the arduino
        # steering, throttle, engage_killswitch, engage_autonomous_driving, reset_odo = process_arduino_inputs()


        # Check for the engage auto button
        if engage_autonomous_driving:
            dm.print_info("Switching to autonomous mode")
            is_autonomous = True
        
        # Check for killswitch
        if engage_killswitch:
            dm.print_warning("Killswitch engaged")
            is_autonomous = False


        # Branch for training data, teleop,  and autonomous
        if record:
            log_data(steering, throttle, logging_dict, "manual")
        else:
            if is_autonomous:
                drive_autonomously(sess, net_model, car_port, logging_dict)
            else:
                drive_manually(steering, throttle, car_port, logging_dict)
        
        # Display for debug
        # cv2.imshow("frame", logging_dict["frame"])
        # cv2.waitKey(1)

        # Attempt to go at 30 fps. In reality, we could go slower if something hiccups.
        seconds = time.time() - loop_start_time
        dm.print_debug("seconds: {}".format(seconds))

        while seconds < 1 / 30.:
            time.sleep(0.001)
            seconds = time.time() - loop_start_time
        
        # Increment the frame_counter
        logging_dict["frame_count"] += 1

        # Update the time, in ms
        logging_dict["milliseconds"] = time.time() * 1000.0



    
# import math
# import os
# import re
# import sys
# import time
# import subprocess

# import cv2
# from docopt import docopt
# import numpy as np
# import platform
# import serial
# #import tensorflow as tf

# import camera
# import key_watcher

# import manual_throttle_map

# # Kartputer modules
# main_car_directory = os.path.dirname(os.path.realpath(__file__))
# carputer_directory = os.path.dirname(main_car_directory)
# nn_directory = os.path.join(carputer_directory, "NeuralNet")

# # Configuration file
# sys.path.append(carputer_directory)
# import config



# # Get args.
# args = docopt(__doc__)


# # Check the mode: recording vs TF driving vs TF driving + recording.
# if args['record']:
#     we_are_autonomous = False
#     we_are_recording = True
#     print("\n------ Ready to record training data ------\n")
# elif args['tf']:
#     we_are_autonomous = True
#     we_are_recording = True
#     print("\n****** READY TO DRIVE BY NEURAL NET and record data ******\n")


# # Set up camera and key watcher.
# camera_stream = camera.CameraStream(src=config.camera_id).start()
# last_key = ['']
# key_watcher.KeyWatcher(last_key).start()


# # Setup buffers and vars used by arduinos.
# buffer_in = ''
# buffer_out = ''
# milliseconds = 0.0
# last_odometer_reset = 0
# odometer_ticks = 0
# button_arduino_out = 0
# button_arduino_in = 0


# def setup_serial_and_reset_arduinos():
#     # This will set up the serial ports. If they are already set up, it will
#     # reset them, which also resets the Arduinos.
#     print("Setting up serial and resetting Arduinos.")
#     # On MacOS, you can find your Arduino via Terminal with
#     # ls /dev/tty.*
#     # then you can read that serial port using the screen command, like this
#     # screen /dev/tty.[yourSerialPortName] [yourBaudRate]
#     if os.name == 'nt':
#         name_in = 'COM3'
#         name_out = 'COM4'
#     else:
#         name_in = '/dev/tty.usbmodem14231'  # 5v Arduino Uno (16 bit)
#         name_out = '/dev/tty.usbmodem14221'  # 3.3v Arduino Due (32 bit)
#     # 5 volt Arduino Duemilanove, radio controller for input.
#     port_in = serial.Serial(name_in, 38400, timeout=0.0)
#     # 3 volt Arduino Due, servos for output.
#     port_out = serial.Serial(name_out, 38400, timeout=0.0)
#     # Flush for good luck. Not sure if this does anything. :)
#     port_in.flush()
#     port_out.flush()
#     print("Serial setup complete.")config.manual_driving_log_dir, config.auto_driving_log_dir
#     return port_in, port_out


# def make_data_folder(base_path):
#     # Make a new dir to store data.
#     base_path = os.path.expanduser(base_path)
#     session_dir_name = time.strftime('%Y_%m_%d__%H_%M_%S_%p')
#     session_full_path = os.path.join(base_path, session_dir_name)
#     if not os.path.exists(session_full_path):
#         os.makedirs(session_full_path)
#     return session_full_path


# def process_input(port_in, port_out):
#     """Reads steering, throttle, aux1 and button data reported from the arduinos.

#     Returns: (steering, throttle, button_arduino_in, button_arduino_out)

#     Return values may be None if the data from the arduino isn't related to the
#     steering or throttle.
#     """
#     # Input is buffered because sometimes partial lines are read
#     global button_arduino_in, button_arduino_out, buffer_in, buffer_out, odometer_ticks, milliseconds
#     try:
#         buffer_in += port_in.read(port_in.in_waiting).decode('ascii')
#         buffer_out += port_out.read(port_out.in_waiting).decode('ascii')
#     except UnicodeDecodeError:
#         # We can rarely get bad data over the serial port. The error looks like this:
#         # buffer_in += port_in.read(port_in.in_waiting).decode('ascii')
#         # UnicodeDecodeError: 'ascii' codec can't decode byte 0xf0 in position 0: ordinal not in range(128)
#         buffer_in = ''
#         buffer_out = ''
#         print("Mysterious serial port error. Let's pretend it didn't happen. :)")
#     # Init steering, throttle and aux1.
#     steering, throttle, aux1 = None, None, None
#     # Read lines from input Arduino
#     while '\n' in buffer_in:
#         line, buffer_in = buffer_in.split('\n', 1)
#         match = re.search(r'(\d+) (\d+) (\d+)', line)
#         if match:
#             steering = int(match.group(1))
#             throttle = int(match.group(2))
#             aux1 = int(match.group(3))
#         if line[0:1] == 'S':
#             # This is just a toggle button
#             button_arduino_in = 1 - button_arduino_in
#             print "ButtonAIn toggle"
#     # Read lines from output Arduino
#     while '\n' in buffer_out:
#         line, buffer_out = buffer_out.split('\n', 1)
#         if line[0:3] == 'Mil':
#             sp = line.split('\t')
#             milliseconds = int(sp[1])
#             odometer_ticks += 1
#         if line[0:6] == 'Button':
#             sp = line.split('\t')
#             button_arduino_out = int(sp[1])
#     return steering, throttle, aux1, button_arduino_in, button_arduino_out


# def process_output(old_steering, old_throttle, steering, throttle, port_out):
#     # Adjust the steering and throttle.
#     throttle = 90 if 88 <= throttle <= 92 else min(throttle, 110)
#     # Update steering
#     if old_steering != steering:
#         port_out.write(('S%d\n' % steering).encode('ascii'))
#     # Update throttle
#     if old_throttle != throttle:
#         port_out.write(('D%d\n' % throttle).encode('ascii'))
#     # Send keepalive.
#     port_out.write(('keepalive\n').encode('ascii'))
#     # Write all.
#     port_out.flush()


# def invert_log_bucket(a):
#     # Reverse the function that buckets the steering for neural net output.
#     # This is half in filemash.py and a bit in convnet02.py (maybe I should fix)
#     # steers[-1] -= 90
#     # log_steer = math.copysign( math.log(abs(steers[-1])+1, 2.0) , steers[-1]) # 0  -> 0, 1  -> 1, -1 -> -1, 2  -> 1.58, -2 -> -1.58, 3  -> 2
#     # gtVal = gtArray[i] + 7
#     steer = a - 7
#     original = steer
#     steer = abs(steer)
#     steer = math.pow(2.0, steer)
#     steer -= 1.0
#     steer = math.copysign(steer, original)
#     steer += 90.0
#     steer = max(0, min(179, steer))
#     return steer


# ##########################
# # Tensorflow Functions   #
# ##########################
# def setup_tensorflow():
#     """Restores a tensorflow session and returns it if successful
#     """
#     net_model = NNModel()

#     tf_config = tf.ConfigProto(device_count = {'GPU':config.should_use_gpu})
#     sess = tf.Session(config=tf_config)

#     # Add ops to save and restore all of the variables
#     saver = tf.train.Saver()

#     # Load the model checkpoint file
#     try:
#         tmp_file = config.tf_checkpoint_file
#         print("Loading model from config: {}".format(tmp_file))
#     except:
#         tmp_file = config.load('last_tf_model') #gets the cached last tf trained model
#         print "loading latest trained model: " + str(tmp_file)
#         # print("CAN'T FIND THE GOOD MODEL")
#         # sys.exit(-1)

#     # Try to restore a session
#     try:
#         saver.restore(sess, tmp_file)
#     except:
#         print("Error restoring TF model: {}".format(tmp_file))
#         # sys.exit(-1)

#     return sess, net_model

# def do_tensorflow(sess, net_model, frame, odo_ticks, vel):
#     # Resize our image from the car
#     resized = cv2.resize(frame, (128, 128))
#     assert resized.shape == (128, 128, 3)  # Must be correct size and RGB, not RGBA.

#     # speed = logging_dict["speedometer"]

#     # Setup the data and run tensorflow
#     batch = TrainingData.FromRealLife(resized, odo_ticks, vel)
#     [steer_regression, throttle_regression] = sess.run([net_model.steering_regress_result, net_model.throttle_pred], feed_dict=batch.FeedDict(net_model))
#     steer_regression += 90
#     throttle_regression += 90

#     # Get to potentiometer
#     # steer_regression = config.TensorflowToSteering(steer_regression)

#     # Map to what car wants
#     # throttle = invert_log_bucket(throttle_pred)

#     return steer_regression, throttle_regression

# # # TOGGLE THIS TO RUN TENSORFLOW
# # if True:
# #     x, odo, vel, pulse, steering_, throttle_, keep_prob, train_mode, train_step, steering_pred, steering_accuracy, throttle_pred, throttle_accuracy, steering_softmax, throttle_softmax, pulse_softmax, conv_maxes, debug_layer = convshared.gen_graph_ops()
# #     sess = tf.Session()
# #     # Add ops to save and restore all the variables.
# #     saver = tf.train.Saver()
# #     # tempfile = args['<path-to-model>'] #+ "/model.ckpt"
# #     # tempfile = "/Users/otaviogood/sfe-models/first-run-0937.ckpt"
# #     #tempfile = "/Users/otaviogood/sfe-models/golden/turning-bias-cleaned-data-0654.ckpt"
# #     # tempfile = "/Users/otaviogood/sfe-models/last-model-1213.ckpt"
# #     # tempfile = "/Users/otaviogood/sfe-models/last-last-model-1251.ckpt"
# #     try:
# #         tempfile = config.tf_checkpoint_file #try to load the hardcoded config file path
# #         print "loading model from config: " + str(tempfile)
# #     except:
# #         tempfile = config.load('last_tf_model') #gets the cached last tf trained model
# #         print "loading latest trained model: " + str(tempfile)
# #
# #     try:
# #         saver.restore(sess, tempfile)
# #     except:
# #         print 'Cannot restore session'
# #
# #     def do_tensor_flow(frame, odo_relative_to_start, speed):
# #         # Take a camera frame as input, send it to the neural net, and get steering back.
# #         resized = cv2.resize(frame, (128, 128))
# #         assert resized.shape == (128, 128, 3)  # Must be correct size and RGB, not RGBA.
# #         resized = resized.ravel()  # flatten the shape of the tensor.
# #         resized = resized[np.newaxis]  # make a batch of size 1.
# #         # scale values to match what's in filemash.
# #
# #         odo_arr = np.array([odo_relative_to_start / 1000.0])[np.newaxis]
# #         vel_arr = np.array([speed * 10.0])[np.newaxis]
# #         pulse_arr = np.zeros((vel_arr.shape[0], convshared.numPulses))  # HACK HACK!!!
# #         current_odo = odo_relative_to_start / convshared.pulseScale  # scale it so the whole track is in range. MUST MATCH CONV NET!!!
# #         for num in xrange(convshared.numPulses):
# #             # http://thetamath.com/app/y=max(0.0,1-abs((x-2)))
# #             pulse_arr[0, num] = max(0.0, 1 - abs(current_odo - num))
# #         steering_result, throttle_result = sess.run([steering_pred, throttle_pred], feed_dict={x: resized, keep_prob: 1.0, odo: odo_arr, vel: vel_arr, pulse: pulse_arr, train_mode: 0.0})  # run tensorflow
# #         steer = invert_log_bucket(steering_result[0])
# #         if config.use_throttle_manual_map:
# #             throt = manual_throttle_map.from_throttle_buckets(throttle_result[0])
# #         else:
# #             throt = invert_log_bucket(throttle_result[0])
# #         return (steer, throt)
# #
# #
# # if False:
# #     ops = model.network.graph_ops(model.network.params)
# #     x_input = ops[0]
# #     steering_output = ops[3]
# #     throttle_output = ops[4]
# #     sess = tf.Session()
# #     saver = tf.train.Saver()
# #     saver.restore(sess, "model/model.cpkt")
# #
# #     def do_tensor_flow(frame, odo, speed):
# #         # Take a camera frame as input, send it to the neural net, and get steering back.
# #         resized = cv2.resize(frame, (128, 128))
# #         assert resized.shape == (128, 128, 3)  # Must be correct size and RGB, not RGBA.
# #         resized = resized[np.newaxis]  # make a batch of size 1.
# #         steering, throttle = sess.run(
# #             [steering_output, throttle_output], feed_dict={x_input: resized})  # run tensorflow
# #         steering = int(steering[0][0] + 90)
# #         throttle = int(throttle[0][0] + 90)
# #         return steering, throttle

# # This checks that we are running the program that allows us to close the lid of our mac and keep running.
# def check_for_insomnia():
#     print("Checking for Insomnia (necessary for everything to work during lid close)")
#     proc = subprocess.Popen(["ps aux"], stdout=subprocess.PIPE, shell=True)
#     (out, err) = proc.communicate()

#     if not "Insomnia" in out:
#         print "\nERROR: YOU ARE NOT RUNNING InsomniaX."
#         print "THAT IS THE PROGRAM THAT LETS YOU SHUT THE LID ON THE MAC AND KEEP IT RUNNING."
#         print "How are you gonna drive a car if your driver is asleep?"
#         sys.exit(0)

# def main():
#     global last_odometer_reset
#     # Init some vars..
#     old_steering = 0
#     old_throttle = 0
#     old_aux1 = 0
#     steering = 90
#     throttle = 90
#     aux1 = 0
#     frame_count = 0
#     session_full_path = ''
#     last_switch = 0
#     button_arduino_out = 0
#     currently_running = False
#     override_autonomous_control = False
#     train_on_this_image = True
#     vel = 0.0
#     last_odo = 0
#     last_millis = 0.0

#     # Check for insomnia
#     if platform.system() == "Darwin":
#         check_for_insomnia()

#     # Setup ports.
#     port_in, port_out = setup_serial_and_reset_arduinos()

#     # Setup tensorflow
#     sess, net_model = setup_tensorflow()

#     # Start the clock.
#     drive_start_time = time.time()
#     print 'Awaiting switch flip..'

#     while True:
#         loop_start_time = time.time()

#         # Switch was just flipped.
#         if last_switch != button_arduino_out:
#             last_switch = button_arduino_out
#             # See if the car started up with the switch already flipped.
#             if time.time() - drive_start_time < 1:
#                 print 'Error: start switch in the wrong position.'
#                 sys.exit()

#             if button_arduino_out == 1:
#                 currently_running = True
#                 print '%s: Switch flipped.' % frame_count
#                 last_odometer_reset = odometer_ticks
#                 if we_are_recording and (not we_are_autonomous):
#                     session_full_path = make_data_folder('~/training-images')
#                     print 'STARTING TO RECORD.'
#                     print 'Folder: %s' % session_full_path
#                     config.store('last_record_dir', session_full_path)
#                 elif we_are_recording and we_are_autonomous:
#                     session_full_path = make_data_folder('~/tf-driving-images')
#                     print 'DRIVING AUTONOMOUSLY and STARTING TO RECORD'
#                     print 'Folder: %s' % session_full_path
#                 else:
#                     print("DRIVING AUTONOMOUSLY (not recording).")
#             else:
#                 print("%s: Switch flipped. Recording stopped." % frame_count)
#                 currently_running = False

#         # Read input data from arduinos.
#         new_steering, new_throttle, new_aux1, button_arduino_in, button_arduino_out = (
#             process_input(port_in, port_out))
#         if new_steering != None:
#             steering = new_steering
#         if new_throttle != None:
#             throttle = new_throttle
#         if new_aux1 != None:
#             aux1 = new_aux1

#         # Check to see if we should stop the car via the RC during TF control.
#         # But also provide a way to re-engage autonomous control after an override.
#         if we_are_autonomous and currently_running:
#             if (steering > 130 or steering < 50) and throttle > 130:
#                 if not override_autonomous_control:
#                     print '%s: Detected RC override: stopping.' % frame_count
#                     override_autonomous_control = True
#             if abs(aux1 - old_aux1) > 400 and override_autonomous_control:
#                 old_aux1 = aux1
#                 print '%s: Detected RC input: re-engaging autonomous control.' % frame_count
#                 override_autonomous_control = False

#         # Check to see if we should reset the odometer via aux1 during manual
#         # driving. This is Button E on the RC transmitter.
#         # The values will swing from ~1100 to ~1900.
#         if abs(aux1 - old_aux1) > 400:
#             old_aux1 = aux1
#             print '%s: Resetting the odometer.' % frame_count
#             last_odometer_reset = odometer_ticks

#         # Overwrite steering with neural net output in autonomous mode.
#         # This seems to take about 10ms.
#         if we_are_autonomous and currently_running:
#             # Calculate velocity from odometer. Gets weird when stopped.
#             if odometer_ticks != last_odo and milliseconds > last_millis:
#                 vel = (float(odometer_ticks) - last_odo) / (milliseconds - last_millis)
#                 if last_millis == 0 and last_odo == 0:
#                     vel = 0
#                 if odometer_ticks - last_odo > 50 or last_odo >= odometer_ticks:
#                     vel = 0
#                 last_odo = odometer_ticks
#                 last_millis = milliseconds
#             # Read a frame from the camera.
#             frame = camera_stream.read()
#             steering, throttle = do_tensorflow(sess, net_model, frame, odometer_ticks - last_odometer_reset, vel)
#             # steering, throttle = do_tensor_flow(frame, odometer_ticks - last_odometer_reset, vel)

#         if we_are_recording and currently_running:
#             # TODO(matt): also record vel in filename for tf?
#             # Read a frame from the camera.
#             frame = camera_stream.read()
#             # Save image with car data in filename.
#             cv2.imwrite("%s/" % session_full_path +
#                 "frame_" + str(frame_count).zfill(5) +
#                 "_thr_" + str(throttle) +
#                 "_ste_" + str(steering) +
#                 "_mil_" + str(milliseconds) +
#                 "_odo_" + str(odometer_ticks - last_odometer_reset).zfill(5) +
#                 ".png", frame)
#         else:
#             frame = camera_stream.read()
#             cv2.imwrite('/tmp/test.png', frame)

#         if override_autonomous_control:
#             # Full brake and neutral steering.
#             throttle, steering = 0, 90

#         # Send output data to arduinos.
#         process_output(old_steering, old_throttle, steering, throttle, port_out)
#         old_steering = steering
#         old_throttle = throttle

#         # Attempt to go at 30 fps. In reality, we could go slower if something hiccups.
#         seconds = time.time() - loop_start_time
#         while seconds < 1 / 30.:
#             time.sleep(0.001)
#             seconds = time.time() - loop_start_time
#         frame_count += 1


if __name__ == '__main__':
    main()
