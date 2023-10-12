import argparse

import lmfit as lmfit
import pandas as pd
from scripts.util import *


def func_log(x, n, a):
    """Return values from a general log function."""
    return -n * np.log10(x) - a


def signal_profiling(_filename, graph_or_not):
    # open file and read RSSI signal
    step_meters = 4.5 # 2.5
    # for original experiment 4.5 for simulation 1:1 and 2.5 for inferred walking speed latter experiments
    _signal, _distance, file = prepare_signal(_filename)

    # find steps
    _steps = np.array(find_steps(_distance.values, meters_per_step=step_meters))
    # Apply median filter to raw data in discrete steps
    signal_mean = function_per_step(file, mean)
    # Apply kalman filter to raw data in discrete steps
    # _signal_kalman = function_per_step(file, kalman)
    # Fit Logarithmic curve of signal loss to median and find c and n
    _A, _n, _residual = fit_parameters(_steps, signal_mean)
    # Find the predicted y's for each of our steps
    _log_of_distance_discrete = distance_to_rssi(_steps, _A, _n)
    # Find the predicted y's for each of all our dataset of distances (non discrete)
    _log_of_distance_raw = distance_to_rssi(_distance * step_meters, _A, _n)
    # Inverse of logarithmic function to visualize distance over signal median, make it lineal
    _distance_infered_mean = rssi_to_distance(signal_mean, _A, _n)
    # Inverse of logarithmic function to visualize distance over signal, make it lineal
    _distance_infered = rssi_to_distance(_signal, _A, _n)
    # Graph it all!!
    if graph_or_not:
        plot_signals([_signal], [filename, 'Signal Kalman'], xlabel="Meters",
                     ylabel="Signal",
                     title="RSSI vs Distance",
                     xi=_distance * step_meters)
        plot_signals([signal_mean, _log_of_distance_discrete], [filename, 'log_regression'],
                     title="RSSI Mean and Log fit vs Distance "
                           f"A= {str(round(_A))}"
                           f" N= {str(round(_n))}"
                           f" R%={str(round(_residual))}",
                     xlabel="Meters",
                     ylabel="Signal Mean",
                     xi=_steps)
        plot_signals([_distance_infered, rssi_to_distance(_log_of_distance_raw, _A, _n)],
                     [filename, 'log_regression'],
                     xlabel="Step Measurement", ylabel="Predicted Distance",
                     title="Inverse log, "
                           f"A= {str(round(_A))}"
                           f" N= {str(round(_n))}"
                           f" R%={str(round(_residual))}"
                     , xi=_distance * step_meters)
        plot_signals([_distance_infered_mean, rssi_to_distance(_log_of_distance_discrete, _A, _n)],
                     [filename, "log_regression"], xlabel="Step Measurement",
                     ylabel="Distance", title="Inverse Log Mean A=" + str(round(
                _A)) + f" N= {str(round(_n))}" + " R%= " + str(round(_residual)), xi=_distance.unique() * step_meters)
    log_canonical = rssi_to_distance(range(1, 50), _A, _n)
    return signal_mean, _steps, _log_of_distance_discrete, _A, _n, _residual, log_canonical


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename',
                        default='data/2022-04-26/192.168.4.2.csv,data/2022-04-26/192.168.4.3.csv,'
                                + 'data/2022-04-26/192.168.4.4.csv,data/2022-04-26/192.168.4.6.csv,'
                                + 'data/2022-04-26/192.168.4.7.csv,data/2022-04-26/192.168.4.8.csv,'
                                  'data/2022-04-26/192.168.4.9.csv')
    parser.add_argument('--verbose', nargs='?', help='data filename',
                        default=False)
    args = parser.parse_args()
    filenames = str(args.file).split(",")
    logs = []
    signals = []
    logs_canonical = []
    print(filenames)
    for filename in filenames:
        signal_mean, _steps, log_of_distance_discrete, _A, _n, _residual, log_canonical = signal_profiling(
            filename, args.verbose)
        signals.append(signal_mean)
        logs.append(log_of_distance_discrete)
        logs_canonical.append(log_canonical)
        plot_signals([signal_mean, log_of_distance_discrete], [filename, 'log_regression'],
                     title=f"RSSI Mean and Log fit vs Distance  "
                           f"A= {str(round(_A))}"
                           f" N= {str(round(_n))}"
                           f" R%={str(round(_residual))}",
                     xlabel="Meters",
                     ylabel="Signal Mean",
                     xi=_steps)

    if len(filenames) > 1:
        plot_signals(signals, labels=filenames, title="Signal mean of every device",
                     xlabel="Meters",
                     ylabel="Signal Mean",
                     xi=_steps
                     )
        plot_signals(logs, labels=filenames, title="Logs applied to each distance captured",
                     xlabel="Meters",
                     ylabel="Signal Mean",
                     xi=_steps
                     )
#        plot_signals(logs_canonical, labels=filenames, title="Log canonical",
#                     xlabel="x",
#                     ylabel="Log"
#                     )
