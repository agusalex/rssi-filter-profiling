import argparse
from scripts.util import *


def signal_analyzer(_filename):
    _signal, _, file = prepare_signal(_filename)
    # Apply median filter to raw data in discrete steps
    _signal_median = function_per_step(file, mean)
    _signal_kalman = KalmanFilter().kalman_filter(_signal_median)
    plot_signals([_signal], [_filename, 'Signal'],
                 ylabel="Signal",
                 title="Signal",
                 )
    plot_signals([_signal_median], [_filename, 'Signal Mean'],
                 ylabel="Signal Mean",
                 title="Signal Mean",
                 )
    plot_signals([KalmanFilter().kalman_filter(_signal_kalman)], [_filename, 'Signal Kalman'],
                 ylabel="Signal Mean + Kalman",
                 title="Signal Mean + Kalman",
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filtering strategies for rssi time series')
    parser.add_argument('--file', nargs='?', help='data filename',
                        default='data/2022-05-16/tri-1/11.csv')
    args = parser.parse_args()
    filenames = str(args.file).split(",")
    for f in filenames:
        signal_analyzer(f)
