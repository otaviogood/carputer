
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
0. train a model: `python NeuralNet/convnet02.py`. Train for minimum 1500 iterations, ideally around 5000 iterations.
0. use this model to drive the car (see above)


## Analysis

* for training info, see `debug.html` -- reminder: < 7 is right, > 7 is left
* run `analysis/make_video.py` for debug videos
* use `analysis/plot_vs_time.py` to view telemetry data


### Hardware TODOs

- [ ] Fix the radio control dropped signal error
- [ ] Get the TX1 working
- [ ] Get the IMU recording data


### Software TODOs

- [ ] Fix keepalive on Arduino
- [ ] Look into remote SSH type software so we don't have to keep popping open the car.

### Hardware setup
Updates - We no longer use the IMUs and we're no longer trying to run the NVidia TX1 computer. Macbook works better.
![Wiring diagram](https://github.com/otaviogood/carputer/blob/master/CarDiagram.jpg "Wiring diagram")

## Simulator

Work in progress, of course. The simulator runs in Unity. I didn't check in the lighting files because they are big, but if you build lighting, it should look like this...
![Unity sim](https://github.com/otaviogood/carputer/blob/master/warehouse_sim.jpg "Unity sim")


## Nvidia TX2 Pipeline
In the "newer" version of the car we replaced the MacBook Pro with an Nvidia TX2. Here is the current way to get the car up and running (with all the weird quirks)

### TX2 Helpers
The TX2 has a few helper aliases on the vehicle. They are located in the `~/.bashrc` and are able to be called anywhere. Here they are:

```
alias record="nohup python2 /home/nvidia/workspace/carputer/main_car.py record & tail -f nohup.out"
alias tf="nohup python2 /home/nvidia/workspace/carputer/main_car.py tf & tail -f nohup.out"
alias kill-car="pkill python"
```

### Connecting to the vehicle

The vehicle will spin up an access point with SSID `carputer-ap` on boot. It is unprotected.

The TX2 will take the IP address of `10.42.0.1`. Hostname is `nvidia` and the password is `nvidia`. *Note* to drivers at the races: Please don't hack our car.

### Gather Training Data
The training pipeline is relativly the same as described in the README, with a few quality of life improvements. 

To start gathering training data, run the following command

`record`

That will start the `python main_car.py` script with a few extra commands to make sure it keeps running even if we lose connection with the vehicle.


### Train on the vehicle?
You can do that. It is pretty slow, but not too bad. The commands are exactly the same as described in the README. However, one note (and this applies to gathering training data too) is that currently the TX2 has very little hard disk memory (less than 20GB nominally). 

The requires the user to make sure that there is enough space to check if there is enough disk space to write the training images, training npy files, and the models.

I like to use `df -h` or `ncdu` to see where the most disk space is and `rm` accordingly. 

### Run a trained model
Again, the pipeline is pretty much the same with a few minor changes.

Unless you understand how the `dyn_config.json` works, your best bet to load your most recently trained model is to edit the `config.py` file to the model you want to load.

To run, simply type `tf`. That will start up the script to run the car in autonomous mode. 

If at any point you want to stop gathering training data or autonmous data, please run the command `kill-car`. It really just `pkill`s python, but it is nice to just remember that you want to kill the car.


