#!/bin/bash
# this script runs inside the EC2 instance
STAGE=${1:-both}

# run the container script
docker exec $(docker ps -q) aws/container.sh ${STAGE}

# copy the run.out file out (and preprocessor) of the container
mkdir -p output
docker cp $(docker ps -q):/tmp/output/run.out output
docker cp $(docker ps -q):/tmp/output/preprocessor output
