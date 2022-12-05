import copy
import statistics

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scripts.kalman import *
import pandas as pd


def median(step):
    return [statistics.median(step)]


def mean(step):
    return [statistics.mean(step)]


def median_kalman(step):
    return [statistics.median(KalmanFilter().kalman_filter(step))]


def kalman(data):
    return KalmanFilter().kalman_filter(data)


def create_steps(file, first=None):
    sequence = file['sequence']
    if first is None:
        start = (sequence[0] / 300)
    else:
        start = first / 300
    distance_pre = sequence.apply(lambda x: round(round(x / 300) - start + 1))
    if "node" in file:
        file_edited = pd.DataFrame(
            {"distance": distance_pre, "rssi": file['rssi'], "node": file["node"], "sequence": file["sequence"]})
    else:
        file_edited = pd.DataFrame({"distance": distance_pre, "rssi": file['rssi']})
    _distance = file_edited["distance"]
    file: pd.DataFrame = file_edited
    return file


def prepare_signal(_filename, first=None):
    file: pd.DataFrame = pd.read_csv(_filename)
    _signal = file['rssi']
    if 'sequence' in file:
        file = create_steps(file, first)
    _distance = file["distance"]
    return _signal, _distance, file


def function_per_step(file, function):
    signal_in_steps = cut_signal_in_steps(file['rssi'].values, find_step_cuts(file.values))
    filtered = []
    for step in signal_in_steps:
        for filtered_step in function(step):
            filtered.append(filtered_step)
    return filtered


def apply_filter(row, function):
    filtered = function(row)
    return filtered[0]


def function_per_step_inplace(df: pd.DataFrame, function):
    group_distance = df.groupby(['distance'], as_index=False)['rssi'].apply(function)
    new_df = df.drop(columns="rssi")
    join = group_distance.merge(new_df, how='left', on='distance')
    return join


def cut_signal_in_steps(signal, cuts):
    signals_per_step = []
    for i in range(len(cuts) - 1):
        origin = int(cuts[i])
        to = int(cuts[i + 1])
        signals_per_step.append(signal[origin:to])

    signals_per_step.append(signal[cuts[len(cuts) - 1]:len(signal)])
    return signals_per_step


def find_step_cuts(file):
    cuttings = [0]
    index = file[0][0]
    cut = 0
    if 'node' in file:
        for step, signal, _, _ in file:
            if step != index:
                cuttings.append(cut)
                index = step
            cut += 1
    else:
        for step, signal, in file:
            if step != index:
                cuttings.append(cut)
                index = step
            cut += 1
    return cuttings


def find_steps(distance, meters_per_step=1):
    steps = [distance[0]]
    index = distance[0]
    for step in distance:
        if step != index:
            steps.append(step * meters_per_step)
            index = step
    return steps


def fit_parameters(distance, signal, C=35.5510920, N=29.0735592) -> object:
    initial_guess = dict(c=C, n=N)
    print(int(C), int(N))

    regressor = lmfit.Model(distance_to_rssi)
    results = regressor.fit(signal, x=np.array(distance), **initial_guess, method="slsqp")
    residual = np.linalg.norm(signal - distance_to_rssi(distance,
                                                        results.values['c'],
                                                        results.values['n']))
    print("R = " + str(100 - residual) + " Result" + str(results.values))
    return results.values['c'], results.values['n'], 100 - residual


def log_fit_adaptive(distance, c, n):
    return distance_to_rssi(distance, c, n)


def distance_to_rssi(x, c, n):
    return - n * np.log10(x) - c


def rssi_to_distance(x_values, c, n):
    y_values = []
    for x in x_values:
        y_values.append(10 ** (-1 * (x + c) / n))
    return y_values


def fit(func, x_values, y_values):
    popt, pcov = curve_fit(func, x_values, y_values)
    residual = np.linalg.norm(y_values - func(x_values, *popt))
    return popt, residual


def median_filter(data, w_size=13):
    """ Median filter for 1d data. Edges are handled with a window shift. """
    data2 = copy.deepcopy(data)
    windowed_data = window(data2, w_size)
    w_half_size = w_size // 2
    for i, w_data in enumerate(windowed_data):
        w_data_sort = sorted(w_data)
        data2[i] = w_data_sort[w_half_size]  # zero indexing handles shift to middle
    return data2


def window(data, w_size):
    """ Get all windows for a list with an odd window size. """
    if not getattr(data, '__iter__', False):
        raise ValueError("data must be an iterable")
    if w_size % 2 != 1:
        raise ValueError("window size must be odd")
    if w_size > len(data):
        raise ValueError("window size must be less than the data size")
    w_half_size = w_size // 2
    windows = []
    for i, _ in enumerate(data):
        if i < w_half_size:
            win = data[0:w_size]
            windows.append(win)
            continue
        if i > len(data) - 1 - w_half_size:
            win = data[-w_size:]
            windows.append(win)
            continue
        win = data[i - w_half_size:i + w_half_size + 1]
        windows.append(win)
    return windows


def plot_signals(yi, labels, ylabel="Y", xlabel="X", title="RSSI over Distance", xi=None, ):
    """

    Auxiliary function for plotting

    input:
        - labels: labels of input signals

    output:
        - display plot

    """

    if xi is None:
        xi = range(len(yi[0]))
    alphas = [1, 0.45, 0.45, 0.45, 0.45]  # just some opacity values to facilitate visualization

    plt.figure()

    for j, y in enumerate(yi):  # iterates on all signals
        plt.plot(xi, y, '-o', label=labels[j], markersize=2, alpha=alphas[j])

    plt.grid()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

    return
