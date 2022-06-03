#!/usr/bin/env bash

# find container tag from L4T version
source docker/tag.sh

# push image
push() 
{
	sudo docker rmi $CONTAINER_REMOTE_IMAGE
	sudo docker tag $1 $CONTAINER_REMOTE_IMAGE
	
	echo "pushing image $CONTAINER_REMOTE_IMAGE"
	sudo docker push $CONTAINER_REMOTE_IMAGE
	echo "done pushing image $CONTAINER_REMOTE_IMAGE"
}

push $TAG