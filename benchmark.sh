#!/bin/bash

# clear tensorboard log files
rm logs/*

# restart tensorboard
killall tensorboard
tensorboard --logdir=./logs &

# get a timestamped filename
NOW=$(date +"%Y%m%d_%H%M%S")
FILE="results/benchmark.$NOW.txt"
echo "writing to $FILE"

# run the project
python -u capstone.py config/benchmark.json $NOW | tee $FILE

