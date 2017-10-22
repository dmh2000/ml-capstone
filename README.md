# ml-capstone
machine learning capstone

## Running on Linux

- select or create a json config file
    - preconfigured files are in config/model[n]
- \>./model.sh [config file]
- the script creates a timestamp
- information is printed to console and are stored in 'results/timestamp'


## Running on Windows

- select or create a json config file
    - preconfigured files are in config/model[n] 
- \>model.cmd [config filename] [optional timestamp or label]
- the windows cmd file doesn't create a timestamp for you
- information is printed to console and are stored in 'results/timestamp'


## Python Files

### capstone.py

This is the main file that is run by the shell scripts. 
This file does all the data preprocessing and execution
of the model, except for defining the model itself. see 
lib/models.py

### lib/models.py

This file contains all the models used in the testing. It
passes a dictionary to the main program with keys as
names of the models. 


### lib/srtm.py

Utility functions for reading and processing the SRTM data files


### lib/metrics.py

Utility functions for printing and plotting metrics

## JSON Config Files

a valid json config file has the following keys
and attributes:
<pre>
{
  "model"     : "benchmark",
  "datafile"  : "data/level1/N39W120.hgt",
  "divisor"   : "16",
  "augments"  : "15",
  "epochs"    : "300"
}
</pre>

- "model" is used to select the desired model from the dictionary in models.py
- "datafile" specifies the SRTM data file to use as input
- "divisor" specifies the row/column division values for segmenting the SRTM data into subimages
- "augments" specifies the number of ImageDataGen augments to create.
It is useful to specify a (multiple of 10) - 1 which makes
the train/test/split code avoid fractions
- "epochs" specifies the number of epochs to run the model


## Results Data

When a model is run, information is written to results/[timestamp].
This includes

- information and statistics about the model, including the
keras summary 
- An svg graph of the accuracy and loss over time is created

If tensorboard is running when a model is executed,
it will show up in the tensorboard display
 
