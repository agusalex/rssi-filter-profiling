import argparse

import lmfit as lmfit
import pandas as pd
from scripts.util import *


def func_log(x, a, c):
    """Return values from a general log function."""
    return -a * np.log10(x) - c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename', default='data/simulation.csv')

    args = parser.parse_args()

    _filename = args.file
    step_meters = 1
    # open file and read RSSI signal

    _file = pd.read_csv(_filename)
    _signal = _file['rssi']
    _distance = _file['distance']
    # find steps
    _steps = np.array(find_steps(_distance.values, meters_per_step=step_meters))
    # Apply median filter to raw data in discrete steps
    _signal_median = function_per_step(_file, median)
    # Apply kalman filter to raw data in discrete steps
    _signal_kalman = function_per_step(_file, kalman)

    # Fit Logarithmic curve of signal loss to median and find c and n
    _C, _n, _b, _residual = find_coeficient(_steps, _signal_median)

    # Find the predicted y's for each of our steps
    _log_of_distance_discrete = distance_to_rssi(_steps, _C, _n, _b)
    # Find the predicted y's for each of all our dataset of distances (non discrete)
    _log_of_distance_raw = distance_to_rssi(_distance * step_meters, _C, _n, _b)
    # Inverse of logarithmic function to visualize distance over signal median, make it lineal
    _distance_infered_median = rssi_to_distance(_signal_median, _C, _n, _b)
    # Inverse of logarithmic function to visualize distance over signal, make it lineal
    _distance_infered = rssi_to_distance(_signal, _C, _n, _b)

    # Graph it All!

    # plot_signals([_signal, _signal_kalman], [_filename, 'Kalman_filter'], xlabel="Meters", ylabel="Signal",
    #              title="Signal vs Distance",
    #              xi=_distance * step_meters)

    plot_signals([_signal_kalman, _log_of_distance_raw], [_filename, 'log_regression'], xlabel="Meters",
                 ylabel="Signal",
                 title="Signal vs Distance C=" +
                       str(round(_C)) + " N= " + str(round(_n)) + " B= " + str(round(_b)) + " R%= " + str(
                     round(_residual)),
                 xi=_distance * step_meters)

    plot_signals([_signal_median, _log_of_distance_discrete], [_filename, 'log_regression'], title="Signal Median vs "
                                                                                                   "Distance  C=" + str(
        round(_C)) + " N= " + str(round(_n)) + " B= " + str(round(_b)) + " R%= " + str(round(_residual)), xi=_steps)

    plot_signals([_distance_infered, rssi_to_distance(_log_of_distance_raw, _C, _n, _b)],
                 [_filename, 'log_regression'],
                 xlabel="Step Measurement", ylabel="Predicted Distance",
                 title="Inverse log, C=" + str(round(_C)) + " N= " + str(round(_n))
                       + " B= " + str(round(_b)) + " R%= " + str(round(_residual)), xi=_distance)

    plot_signals([_distance_infered_median, rssi_to_distance(_log_of_distance_discrete, _C, _n, _b)],
                 [_filename, "log_regression"], xlabel="Step Measurement",
                 ylabel="Distance", title="Inverse Median C=" + str(round(
            _C)) + " N= " + str(round(_n)) + " B= " + str(round(_b)) + " R%= " + str(round(_residual)))
