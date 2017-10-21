#!/bin/bash

# clear tensorboard log files
rm logs/*

# restart tensorboard
killall tensorboard
tensorboard --logdir=./logs &

# get a timestamped filename
NOW=$(date +"%Y%m%d_%H%M%S")
FILE="results/solution.$NOW.txt"
echo "writing to $FILE"

# run the project
python -u capstone.py data/level1/N39W120.hgt 30 29 300 solution  $NOW | tee $FILE
