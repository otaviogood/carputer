# Steps


## Recording pipeline
1. Turn on TX2
1. Plug in USB to TX2
2. Plug in Ethernet to Laptop
3. Find the IP Address of the TX2
  * Currently it defaults to: 10.42.0.94
  * if not, run `sudo nmap -sn 10.42.0.0/24` to find the ip address
  * TODO: Static ip the TX2 for easy scipting and access
4. ssh into the TX2
  * user:password -> nvidia:nvidia
5. Power motors, plug in main battery
5. Run `record`

## Training pipeline
1. ssh into TX2
2. run `kill-car`
3. Copy training data from car to laptop
  * Training images are timestamped in ~/training-images
  * From laptop `scp -r nvidia@10.42.0.94:/home/nvidia/training-images/[data] [wherever on your laptop]`
  * TODO: This can be scripted
4. cd into carputer directory
5. `python NeuralNet/filemash.py [path to training images]`
6. `python NeuralNet/filemash.py [path to test images] --gen_test`
7. `python NeuralNet/convnet02.py`

## Deployment pipeline
1. Find your latest trained model in ~/convnet02-results/
2. `scp -r ~/convnet02-results/[model directory] nvidia@10.42.0.94:/home/nvidia/convnet02-results/`
  * TODO: This can be scriptable. Copy model to right place
3. ssh into TX2
4. cd ~/workspace/carputer
5. Edit `config.py` and update the path for `tf_checkpoint_file` to point to your new model
   * TODO: This should be automatically updated when you copy the model over
6. Run `tf`

## Other nice to haves
1. A GUI
2. Since we are a little short on HD space on the TX2, cleanup scripts would be nice.
3. The transmitter sometimes drops packets, we should fix this in the Arduino code. A solution would look like if detected dropped packets, maintain the last commanded command

