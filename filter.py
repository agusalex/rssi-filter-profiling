import argparse
import math
import os
import glob
import pandas

from scripts.util import *

# for original experiment 4.5 for simulation 1:1 and 1.2 for inferred walking speed latter experiments (1.2 m/s)
step_meters = 1
# Only for sequence type, not needed for precise measurements (has distance row instead of sequence)
# packets per second group by sequence number of n packets 6.6 for samsung s20 150ms intervals, 6.6 per second
group_by = 6.6


def signal_analyzer(_filename):
    _signal, _, file = prepare_signal(_filename, group_by)
    # Apply median filter to raw data in discrete steps
    _signal_median = function_per_step(file, mean)
    _signal_kalman = KalmanFilter().kalman_filter(_signal_median)


def find_first_sequence(files):
    min = math.inf
    for file in files:
        csv = pandas.read_csv(file)
        if 'sequence' in csv:
            csv.sort_values('sequence', inplace=True)
            current = csv['sequence'][0]
            if current < min:
                min = current

    return min


if __name__ == '__main__':
    PATH = r"."
    os.chdir(PATH)
    files = glob.glob("*.csv")
    ignored = ['joined.csv', 'out.csv']
    filtered = list(filter(lambda file: file not in ignored, files))
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename',
                        default=",".join(filtered))
    parser.add_argument('--apply', nargs='?', help='apply inplace',
                        default="n")
    parser.add_argument('--mode', nargs='?', help='apply inplace',
                        default="kalman")
    args = parser.parse_args()
    filenames = str(args.file).split(",")

    first = find_first_sequence(filenames)
    kalmans = []
    for filename in filenames:
        # signal_analyzer(filename)
        signal, distance, df = prepare_signal(filename, group_by, first=first)
        _mean = function_per_step_inplace(df, mean)
        signal_mean = _mean["rssi"]

        _median = function_per_step_inplace(df, median)
        signal_median = _median["rssi"]

        _kalman_signal = KalmanFilter(0.01, 10, ).kalman_filter(df['rssi'])

        #  plot_signals([df['rssi']], [filename, 'Signafl'],
        #                  ylabel="Signal",
        #                 title="Signal",
        #                )

        if args.apply == "n":
            if args.mode == "kalman":
                plot_signals([df['rssi'], _kalman_signal], [filename, 'Signal Kalman'],
                             ylabel="Signal Kalman",
                             title="Signal vs Signal Kalman Filtered ",
                             xlabel="Predicted Distance",
                             xi=distance * step_meters
                             )
            else:
                plot_signals([df['rssi'], _kalman_signal, signal_mean, signal_median],
                             [filename, 'Signal Kalman', 'Signal Mean', 'Signal Median'],
                             ylabel="Signal vs Filtering Methods",
                             xlabel="Predicted Distance",
                             title="Signal vs Filtering Methods",
                             xi=distance * step_meters
                             )
        else:
            if args.mode == "kalman":
                df['rssi'] = _kalman_signal
                # df = df.drop(columns=['distance'], axis=1)
                df.to_csv(filename, encoding='utf-8', index=False)
            else:
                df['rssi'] = signal_mean
                # df = df.drop(columns=['distance'], axis=1)
                df.to_csv(filename, encoding='utf-8', index=False)
