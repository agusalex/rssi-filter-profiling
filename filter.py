import argparse
import math
import os
import glob
import pandas

from scripts.util import *


def signal_analyzer(_filename):
    _signal, _, file = prepare_signal(_filename)
    # Apply median filter to raw data in discrete steps
    _signal_median = function_per_step(file, mean)
    _signal_kalman = KalmanFilter().kalman_filter(_signal_median)


def find_first_sequence(files):
    min = math.inf
    for file in files:
        csv = pandas.read_csv(file)
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
                        default=",".join(filtered))
    args = parser.parse_args()
    filenames: list[str] = str(args.file).split(",")
    first = find_first_sequence(filenames)
    kalmans = []
    for filename in filenames:
        # signal_analyzer(filename)
        _, _, df = prepare_signal(filename, first=first)
        _mean = function_per_step_inplace(df, mean)
        signal_mean = _mean["rssi"]

        _kalman_signal = KalmanFilter(0.01, 10, ).kalman_filter(df['rssi'])

      #  plot_signals([df['rssi']], [filename, 'Signafl'],
   #                  ylabel="Signal",
    #                 title="Signal",
     #                )
        #plot_signals([signal_mean], [filename, 'Signal Mean'],
        #             ylabel="Signal Mean",
        #             title="Signal Mean",
        #             )
        plot_signals([df['rssi'], _kalman_signal], [filename, 'Signal Kalman'],
                     ylabel="Signal Kalman",
                     title="Signal Kalman",
                     )
        df['rssi'] = _kalman_signal
        df = df.drop(columns=['distance'], axis=1)
        if args.apply == "y":
            df.to_csv(filename, encoding='utf-8', index=False)
