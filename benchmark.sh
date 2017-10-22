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


# run the project
python -u capstone.py config/benchmark.json $NOW | tee $FILE

