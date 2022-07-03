import csv

import pandas as pd
import glob
import os

if __name__ == '__main__':
    # setting the path for joining multiple files
    PATH = r"."
    os.chdir(PATH)

    files = glob.glob("*.csv")
    output_name = 'joined.csv'
    ignored = [output_name, 'out.csv']
    filtered = list(filter(lambda file: file not in ignored, files))

    # joining files with concat and read_csv
    df = pd.concat(map(pd.read_csv, filtered), ignore_index=True)
    df.sort_values('sequence', inplace=True)
    df.to_csv(output_name, encoding='utf-8', index=False)
