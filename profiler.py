import argparse

import lmfit as lmfit
import pandas as pd
from scripts.util import *


def func_log(x, a, c):
    """Return values from a general log function."""
    return -a * np.log10(x) - c


def signal_profiling(_filename, graph_or_not):
    # open file and read RSSI signal
    step_meters = 1
    _signal, _distance, file = prepare_signal(_filename)

    # find steps
    _steps = np.array(find_steps(_distance.values, meters_per_step=step_meters))
    # Apply median filter to raw data in discrete steps
    signal_mean = function_per_step(file, mean)
    # Apply kalman filter to raw data in discrete steps
    # _signal_kalman = function_per_step(file, kalman)
    # Fit Logarithmic curve of signal loss to median and find c and n
    _C, _n, _residual = fit_parameters(_steps, signal_mean)
    # Find the predicted y's for each of our steps
    _log_of_distance_discrete = distance_to_rssi_adaptive(_steps, _C, _n)
    # Find the predicted y's for each of all our dataset of distances (non discrete)
    _log_of_distance_raw = distance_to_rssi_adaptive(_distance * step_meters, _C, _n)
    # Inverse of logarithmic function to visualize distance over signal median, make it lineal
    _distance_infered_median = rssi_to_distance_adaptive(signal_mean, _C, _n)
    # Inverse of logarithmic function to visualize distance over signal, make it lineal
    _distance_infered = rssi_to_distance_adaptive(_signal, _C, _n)
    # Graph it all!!
    if graph_or_not:
        plot_signals([_signal], [filenames, 'Signal Kalman'], xlabel="Meters",
                     ylabel="Signal",
                     title="Signal and Signal Kalman vs Distance C=" +
                           str(round(_C)) + " R%= " + str(
                         round(_residual)),
                     xi=_distance)
        plot_signals([signal_mean, _log_of_distance_discrete], [filenames, 'log_regression'],
                     title="Signal Median vs "
                           f"vs Distance  "
                           f"C= {str(round(_C))}"
                           f" N= {str(round(_n))}"
                           f" R%={str(round(_residual))}",
                     xlabel="Meters",
                     ylabel="Signal Median",
                     xi=_steps)
        plot_signals([_distance_infered, rssi_to_distance_adaptive(_log_of_distance_raw, _C, _n)],
                     [filenames, 'log_regression'],
                     xlabel="Step Measurement", ylabel="Predicted Distance",
                     title="Inverse log, C=" + str(round(_C)) + " R%= " + str(round(_residual)), xi=_distance)
        plot_signals([_distance_infered_median, rssi_to_distance_adaptive(_log_of_distance_discrete, _C, _n)],
                     [filenames, "log_regression"], xlabel="Step Measurement",
                     ylabel="Distance", title="Inverse Log Median C=" + str(round(
                _C)) + " R%= " + str(round(_residual)))
    log_canonical = rssi_to_distance_adaptive(range(1, 50), _C, _n)
    return signal_mean, _steps, _log_of_distance_discrete, _C, _n, _residual, log_canonical


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename',
                        default='data/2022-04-26/192.168.4.9.csv')
    args = parser.parse_args()
    filenames = str(args.file).split(",")
    logs = []
    logs_canonical = []
    for filename in filenames:
        signal_median, _steps, log_of_distance_discrete, _C, _n, _residual, log_canonical = signal_profiling(
            filename, True)
        logs.append(log_of_distance_discrete)
        logs_canonical.append(log_canonical)
        plot_signals([signal_median, log_of_distance_discrete], [filename, 'log_regression'],
                     title=f"Signal Median "
                           f"vs Distance  "
                           f"C= {str(round(_C))}"
                           f" N= {str(round(_n))}"
                           f" R%={str(round(_residual))}",
                     xlabel="Meters",
                     ylabel="Signal Median",
                     xi=_steps)

    if len(filenames) > 1:
        plot_signals(logs, labels=filenames, title="Logs applied to each distance captured",
                     xlabel="Meters",
                     ylabel="Signal Median",
                     )

        plot_signals(logs_canonical, labels=filenames, title="Log canonical",
                     xlabel="x",
                     ylabel="Log",
                     )
