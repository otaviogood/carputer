
# Carputer

Carputer is a 1/10th scale self driving car. It drives based on video camera and speedometer inputs. It uses a neural network to drive. The camera image and speedometer get sent to the neural network and the network outputs steering and throttle motor controls for the car.

Since the car uses a neural network for driving, it has to be trained. The training process is basically driving it around a track about 20 times or so using the car's radio control. During that driving, we record all of the radio control inputs along with the video and speedometer data. Once we have that, the neural network can learn how to mimic our driving style by outputting steering and throttle based on the video and speedometer.

The process is to record, train, and then run autonomously, as seen in the steps below.

This is an experimental setup, so it's not super-clean code or hardware.

## Recording pipline
0. Turn on ESC, RC controller. Plug in battery, USB. Start switch to off.
0. Run InsomniaX and disable lid sleep and idle sleep.
0. activate the virtualenv: `source /path/to/venv/bin/activate`
0. run a script to drive and record training data: `python main_car.py record` --
this will let you have manual control over the car
and save out recordings when you flip the switch


## Run autonomously

0. Turn on ESC, RC controller. Plug in battery, USB. Start switch to off.
0. Run InsomniaX and disable lid sleep and idle sleep.
0. activate the virtualenv: `source /path/to/venv/bin/activate`
0. run a script to let tensorflow drive: `python main_car.py tf` --
when you flip the switch, you will lose manual control
and the car will attempt to drive on its own
0. for autonomous kill switch: pull throttle and turn the steering wheel
0. to revive autonomous mode, hit the channel 3 button (near the trigger)


## Training pipline

0. convert TRAINING images to np arrays: `python NeuralNet/filemash.py /path/to/data` (Can be multiple paths)
0. convert TEST images to np arrays: `python NeuralNet/filemash.py /path/to/data --gen_test` (Can be multiple paths)
0. train a model: `python NeuralNet/convnet02.py`
0. use this model to drive the car (see above)


## Analysis

* for training info, see `debug.html` -- reminder: < 7 is right, > 7 is left
* run `analysis/make_video.py` for debug videos
* use `analysis/plot_vs_time.py` to view telemetry data


### Hardware TODOs

- [ ] Make the car drive slower. Smaller wheels? Smaller batteries?


### Software TODOs

- [ ] Fix keepalive on Arduino
- [ ] Debug crowded autonomous runs with lots of people standing around throwing off the neural net
- [ ] Fix the kill switch so it locks brakes instead of coasting. Just a logic bug I think.
- [ ] Look into remote SSH type software so we don't have to keep popping open the car.

### Hardware setup
Updates - We no longer use the IMUs and we're no longer trying to run the NVidia TX1 computer. Macbook works better.
![Wiring diagram](https://github.com/otaviogood/carputer/blob/master/CarDiagram.jpg "Wiring diagram")


## Updated Setup for ESC not saving the mode
0. Plug in LiPo Battery
0. Plug in USB
0. Turn on RC Transmitter
0. Start main_car script
0. Program ESC for race mode
  - TODO: Add details for this
0. Drive

