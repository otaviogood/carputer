
. ec2-spotter/fast_ai/start_spot_no_swap.sh --ami ami-d6ee1dae --securityGroupId sg-543aaf29 --subnetId subnet-a661f3c0

aws ec2 attach-volume --volume-id vol-07775096bc43db288 --instance-id `cat spot_instance_id.txt` --device /dev/sdh

export SSH="ssh -i ~/.ssh/aws-key-fast-ai.pem ubuntu@`cat spot_instance_ip.txt`"
echo $SSH
#

$SSH "mkdir carputer"
$SSH "sudo mount /dev/xvdh carputer"

$SSH "sudo apt-get install python-virtualenv"
$SSH "echo source ~/carputer/venv/bin/activate >> ~/.bashrc"
