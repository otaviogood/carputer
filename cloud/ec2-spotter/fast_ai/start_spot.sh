# The config file was created in ondemand_to_spot.sh
export config_file=my.conf

# Set current dir to working dir - http://stackoverflow.com/a/10348989/277871
cd "$(dirname ${BASH_SOURCE[0]})"

. ../$config_file || exit -1

export request_id=`../ec2spotter-launch $config_file`
echo Spot request ID: $request_id

echo Waiting for spot request to be fulfilled...
aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $request_id  

export instance_id=`aws ec2 describe-spot-instance-requests --spot-instance-request-ids $request_id --query="SpotInstanceRequests[*].InstanceId" --output="text"`

echo Waiting for spot instance to start up...
aws ec2 wait instance-running --instance-ids $instance_id

echo Spot instance ID: $instance_id 

echo 'Please allow the root volume swap script a few minutes to finish.'
if [ "x$ec2spotter_elastic_ip" = "x" ]
then
	# Non elastic IP
	export ip=`aws ec2 describe-instances --instance-ids $instance_id --filter Name=instance-state-name,Values=running --query "Reservations[*].Instances[*].PublicIpAddress" --output=text`
else
	# Elastic IP
	export ip=`aws ec2 describe-addresses --allocation-ids $ec2spotter_elastic_ip --output text --query 'Addresses[0].PublicIp'`
fi	

export name=fast-ai
if [ "$ec2spotter_key_name" = "aws-key-$name" ] 
then
	echo Then connect to your instance: ssh -i ~/.ssh/aws-key-$name.pem ubuntu@$ip
fi
