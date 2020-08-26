#!/usr/bin/env bash

DOCKER_USER=$1

push_retag() 
{
	local src_tag=$1
	local dst_tag=$2
	
	sudo docker rmi $DOCKER_USER/$dst_tag
	sudo docker tag $src_tag $DOCKER_USER/$dst_tag
	
	echo "pushing image $DOCKER_USER/$dst_tag"
	sudo docker push $DOCKER_USER/$dst_tag
}

push() 
{
	push_retag $1 $1
}

# find L4T_VERSION
source tools/l4t-version.sh

# push image
push "jetson-inference:r$L4T_VERSION"