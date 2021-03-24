#! /bin/bash

#####
# docker commands
#
# Usage:
#  ./docker.sh build - build docker image
#  ./docker.sh run - run container interactively
#  ./docker.sh stop - stop current container
#####

# inputs
action=$1

# const
CONTAINER_NAME="deepfakedetection-env"

if [[ $action == "build" ]]; then
    docker build -t deepfakedetection-$(whoami):dev .
elif [[ $action == "run" ]]; then
    docker run --name $CONTAINER_NAME -it deepfakedetection-$(whoami):dev
elif [[ $action == "stop" ]]; then
    docker container stop $CONTAINER_NAME
    docker container rm $CONTAINER_NAME
else
    echo "Unknown command " $action
fi
