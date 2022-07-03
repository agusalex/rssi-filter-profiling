import csv
import math

import pandas as pd
import argparse

if __name__ == '__main__':
    key_index = "sender"
    order_index = "sequence"
    max_cut = 66000  # math.inf
    min_cut = 47008  # 0
    parser = argparse.ArgumentParser(
        description='Split and cut csv capture file ')
    parser.add_argument('--file', nargs='?', help='data filename', default='out.csv')
    parser.add_argument('--split', nargs='?', help='split into multiple files', default='True')
    args = parser.parse_args()
    split = args.split == "True"
    _filename = args.file
    _file = pd.read_csv(_filename)

    files = dict()
    max_row_global = math.inf

    files_matrix = {}
    if not split:
        combined = open("out_filtered" + ".csv", 'w', newline='')

    for base_row in _file.iterrows():
        row = base_row[1]
        id_row = row[key_index]
        if id_row not in files:
            if split:
                new_file = open(id_row + ".csv", 'w', newline='')
            else:
                new_file = combined
            files[id_row] = {'file': csv.writer(new_file),
                             'rows': [],
                             'max_row': 0}

        max_row_for_file = files[id_row]["max_row"]
        order_number = row[order_index]

        # stop when seq is smaller than previous
        if max_row_for_file < order_number:
            # stop when diff of max seq is too big a leap
            if abs(order_number - max_row_for_file) < 100000:
                if max_cut >= order_number >= min_cut:
                    files[id_row]["rows"].append(row)
                    files[id_row]["max_row"] = row[order_index]

    max_ids = set()
    for new_file_data in files.values():
        max_ids.add(new_file_data["max_row"])

    first = True
    for new_file_data in files.values():
        if first or split:
            new_file_data["file"].writerow(["sender", "sequence", "rssi"])
            first = False
        for row in new_file_data["rows"]:
            if row["sequence"] <= min(max_ids):
                new_file_data["file"].writerow(row)
