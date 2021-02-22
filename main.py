import argparse
import pandas as pd
from util import *

parser = argparse.ArgumentParser(
    description='Filtering strategies for rssi time series')
parser.add_argument('--file', nargs='?', help='data filename', default='data/Device1.csv')

args = parser.parse_args()

file_name = args.file
step_meters = 4.5
# open file and read RSSI signal

file = pd.read_csv(file_name)
signal = file['rssi']
distance = file['distance']

# Find Median, Find steps
signal_median = function_per_step(file, median)
steps = np.array(find_steps(distance.values, meters_per_step=step_meters))

# Fit Logarithmic curve of signal loss to median
C = find_coeficient(steps, signal_median)

log_of_distance_discrete = log_fit(steps, C)
log_of_distance_raw = log_fit(distance*step_meters, C)
# Inverse of logarithmic function to visualize distance over signal, make it lineal
distance_infered = rssi_to_distance(signal, C)
distance_infered_median = rssi_to_distance(signal_median, C)

# Graph it All!
plot_signals([signal,log_of_distance_raw], [file_name, 'log_regression'], xlabel="Meters", ylabel="Signal", title="Signal vs Distance C=" + str(C),
             xi=distance * step_meters)

plot_signals([signal_median, log_of_distance_discrete], [file_name, 'log_regression'], title="Signal Median vs Distance  C=" + str(C), xi=steps)

plot_signals([distance_infered, rssi_to_distance(log_of_distance_raw, C)], [file_name, 'log_regression'], xlabel="Step Measurement", ylabel="Predicted Distance",
             title="Inverse log, C=" + str(C),
             xi=distance)

plot_signals([distance_infered_median, rssi_to_distance(log_of_distance_discrete, C)], [file_name, "log_regression"], xlabel="Step Measurement",
             ylabel="Distance", title="Inverse Median C=" + str(C))
