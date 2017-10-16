@echo --- no arguments
python capstone.py
@echo --- one valid argument
python capstone.py data\level2\N39W120.hgt
@echo --- invalid division arguments
python capstone.py data\level2\N39W120.hgt x
@echo --- invalid filename argument
python capstone.py data\level2\N39W120.x 8

