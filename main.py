import argparse
import pandas as pd
from scripts.util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename', default='data/Device1.csv')

    args = parser.parse_args()

    _filename = args.file
    step_meters = 4.5
    # open file and read RSSI signal

    _file = pd.read_csv(_filename)
    _signal = _file['rssi']
    _signal_kalman = function_per_step(_file, kalman)
    _distance = _file['distance']

    # Find Median, Find steps
    _signal_median = function_per_step(_file, median_kalman)
    _steps = np.array(find_steps(_distance.values, meters_per_step=step_meters))

    # Fit Logarithmic curve of signal loss to median
    _C, _residual = find_coeficient(_steps, _signal_median)

    _log_of_distance_discrete = log_fit(_steps, _C)
    _log_of_distance_raw = log_fit(_distance * step_meters, _C)
    # Inverse of logarithmic function to visualize distance over signal, make it lineal
    _distance_infered = rssi_to_distance(_signal_kalman, _C)
    _distance_infered_median = rssi_to_distance(_signal_median, _C)

    # Graph it All!

    plot_signals([_signal, _signal_kalman], [_filename, 'Kalman_filter'], xlabel="Meters", ylabel="Signal",
                 title="Signal vs Distance",
                 xi=_distance * step_meters)

    plot_signals([_signal_kalman, _log_of_distance_raw], [_filename, 'log_regression'], xlabel="Meters",
                 ylabel="Signal",
                 title="Signal vs Distance C=" +
                       str(_C) + " R%= " + str(_residual),
                 xi=_distance * step_meters)

    plot_signals([_signal_median, _log_of_distance_discrete], [_filename, 'log_regression'], title="Signal Median vs "
                 "Distance  C=" + str(_C) + " R%= " + str(_residual),xi=_steps)

    plot_signals([_distance_infered,kalman(_distance_infered), rssi_to_distance(_log_of_distance_raw, _C)], [_filename, "kalman",'log_regression'],
                 xlabel="Step Measurement", ylabel="Predicted Distance",
                 title="Inverse log, C=" +
                       str(_C) + " R%= " + str(_residual),
                 xi=_distance)

    plot_signals([_distance_infered_median, rssi_to_distance(_log_of_distance_discrete, _C)],
                 [_filename, "log_regression"], xlabel="Step Measurement",
                 ylabel="Distance", title="Inverse Median C=" +
                                          str(_C) + " R%= " + str(_residual))
