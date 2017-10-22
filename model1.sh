#!/bin/bash

# clear tensorboard log files
rm logs/*

# restart tensorboard
killall tensorboard
tensorboard --logdir=./logs &

# get a timestamped filename
NOW=$(date +"%Y%m%d_%H%M%S")

# create the results directory
mkdir results/$NOW

# create timestamped filename
FILE="results/$NOW/$NOW.txt"

# print it
echo "writing to $FILE"

# run the project
python -u capstone.py config/level1-N39W120.json $NOW | tee $FILE

