# Median, mean and Kalman Filter plus Log Fit for RSSI to distance prediction

## Getting Started

![first](data/index/output_5_0.png)
![first](data/index/output_5_1.png)
### Project requirements 

First step is to create one and install all necessary project requirements.

    virtualenv env --python=python3
    source env/bin/activate
    pip install -r requirements.txt

## Execution

    python main.py [--file /path/to/file]

You can set the path to a file containing your data, default path is _data/Device.csv_.

For example:

    main.py --file data/sample.csv

## Results
![first](data/index/output_5_2.png)
![first](data/index/output_5_3.png)
![combined2](data/combined2.png)
