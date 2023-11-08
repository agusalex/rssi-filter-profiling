import argparse

import lmfit as lmfit
import pandas as pd
from scripts.util import *

# for original experiment 4.5 for simulation 1:1 and 1.2 for inferred walking speed latter experiments (1.2 m/s)
step_meters = 1
# Only for sequence type measurements, not needed for precise  (has distance row instead of sequence)
# packets per second group by sequence number of n packets 6.6 for samsung s20 150ms intervals, 6.6 per second
group_by = 3.3


def signal_profiling(_filename, graph_or_not):
    _signal, _distance, file = prepare_signal(_filename, group_by)
    # find steps
    _steps = np.array(find_steps(_distance.values))

    # Apply median filter to raw data in discrete steps
    signal_mean = function_per_step(file, mean)
    # Apply kalman filter to raw data in discrete steps
    # _signal_kalman = function_per_step(file, kalman)
    # Fit Logarithmic curve of signal loss to median and find c and n
    _A, _n, _residual = fit_parameters(_steps, signal_mean)
    # Find the predicted y's for each of our steps
    _log_of_distance_discrete = distance_to_rssi(_steps, _A, _n)
    # Find the predicted y's for each of all our dataset of distances (non-discrete)
    _log_of_distance_raw = distance_to_rssi(_distance, _A, _n)
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
                     xi=_distance.unique() * step_meters)
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
    return signal_mean, _distance.unique() * step_meters, _log_of_distance_discrete, _A, _n, _residual, log_canonical


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename',
                        default='192.168.4.2.csv,192.168.4.3.csv,192.168.4.4.csv,192.168.4.6.csv,192.168.4.8.csv,192.168.4.9.csv')

    parser.add_argument('--verbose', nargs='?', help='data filename',
                        default=False)
    args = parser.parse_args()
    filenames = str(args.file).split(",")
    logs = []
    signals = []
    logs_canonical = []
    print(filenames)
    for filename in filenames:
        signal_mean, graph_xi, log_of_distance_discrete, _A, _n, _residual, log_canonical = signal_profiling(
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
                     xi=graph_xi)

    if len(filenames) > 0:
        plot_signals(signals, labels=filenames, title="Signal mean of every device",
                     xlabel="Meters",
                     ylabel="Signal Mean",
                     xi=graph_xi
                     )
        plot_signals(logs, labels=filenames, title="Logs applied to each distance captured",
                     xlabel="Meters",
                     ylabel="Signal Mean",
                     xi=graph_xi
                     )
#        plot_signals(logs_canonical, labels=filenames, title="Log canonical",
#                     xlabel="x",
#                     ylabel="Log"
#                     )
