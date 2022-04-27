import argparse

import lmfit as lmfit
import pandas as pd
from scripts.util import *


def func_log(x, a, c):
    """Return values from a general log function."""
    return -a * np.log10(x) - c


def compute_graph_data(_filename, graph_or_not):
    step_meters = 4
    # open file and read RSSI signal
    file = pd.read_csv(_filename)
    _signal = file['rssi']
    if 'sequence' in file:
        sequence = file['sequence']
        distance_pre = sequence.apply(lambda x: round(x / 300) - (sequence[0] / 300) + 1)
        file_edited = pd.DataFrame({"distance": distance_pre, "rssi": _signal})
        _distance = file_edited["distance"]
        file = file_edited
    else:
        _distance = file["distance"]
    # find steps
    _steps = np.array(find_steps(_distance.values, meters_per_step=step_meters))
    # Apply median filter to raw data in discrete steps
    _signal_median = function_per_step(file, mean)
    # Apply kalman filter to raw data in discrete steps
    # _signal_kalman = function_per_step(file, kalman)
    # Fit Logarithmic curve of signal loss to median and find c and n
    _C, _n, _b, _residual = find_coeficient_adaptive(_steps, _signal_median)
    # Find the predicted y's for each of our steps
    _log_of_distance_discrete = distance_to_rssi_adaptive(_steps, _C, _n, _b)
    # Find the predicted y's for each of all our dataset of distances (non discrete)
    _log_of_distance_raw = distance_to_rssi_adaptive(_distance * step_meters, _C, _n, _b)
    # Inverse of logarithmic function to visualize distance over signal median, make it lineal
    _distance_infered_median = rssi_to_distance_adaptive(_signal_median, _C, _n, _b)
    # Inverse of logarithmic function to visualize distance over signal, make it lineal
    _distance_infered = rssi_to_distance_adaptive(_signal, _C, _n, _b)
    # Graph it all!!
    if graph_or_not:
        plot_signals([_signal], [filenames, 'Signal Kalman'], xlabel="Meters",
                     ylabel="Signal",
                     title="Signal and Signal Kalman vs Distance C=" +
                           str(round(_C)) + " R%= " + str(
                         round(_residual)),
                     xi=_distance)
        plot_signals([_signal_median, _log_of_distance_discrete], [filenames, 'log_regression'],
                     title="Signal Median vs "
                           "Distance  C=" +
                           str(round(_C)) + " R%= " + str(round(_residual)),
                     xlabel="Meters",
                     ylabel="Signal Median",
                     xi=_steps)
        plot_signals([_distance_infered, rssi_to_distance_adaptive(_log_of_distance_raw, _C, _n, _b)],
                     [filenames, 'log_regression'],
                     xlabel="Step Measurement", ylabel="Predicted Distance",
                     title="Inverse log, C=" + str(round(_C)) + " R%= " + str(round(_residual)), xi=_distance)
        plot_signals([_distance_infered_median, rssi_to_distance_adaptive(_log_of_distance_discrete, _C, _n, _b)],
                     [filenames, "log_regression"], xlabel="Step Measurement",
                     ylabel="Distance", title="Inverse Log Median C=" + str(round(
                _C)) + " R%= " + str(round(_residual)))
    log_canonical = rssi_to_distance_adaptive(range(1, 50), _C, _n, _b)
    return _signal_median, _steps, _log_of_distance_discrete, _C, _n, _b, _residual, log_canonical


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename',
                        default='data/2022-04-26/192.168.4.9.csv,'
                                'data/2022-04-26/192.168.4.3.csv,'
                                'data/2022-04-26/192.168.4.4.csv,'
                                'data/2022-04-26/192.168.4.6.csv,'
                                'data/2022-04-26/192.168.4.8.csv')
    args = parser.parse_args()
    filenames = str(args.file).split(",")
    logs = []
    logs_canonical = []
    for filename in filenames:
        signal_median, _steps, log_of_distance_discrete, _C, _n, _b, _residual, log_canonical = compute_graph_data(
            filename, False)
        logs.append(log_of_distance_discrete)
        logs_canonical.append(log_canonical)
        plot_signals([signal_median, log_of_distance_discrete], [filename, 'log_regression'],
                     title=f"Signal Median "
                           f"vs Distance  "
                           f"C= {str(round(_C))}"
                           f" N= {str(round(_n))}"
                           f" B= {str(round(_b))}"
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
