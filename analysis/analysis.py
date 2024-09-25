from typing import Dict
import os
import pandas as pd

'''
This file aggregate the benchmark records from a dir and generate various analytical stats & figures
'''

def generate_csv(df: pd.DataFrame) -> bool:
    pass

def visualize(df: pd.DataFrame) -> bool:
    pass

def process_data(dir_path: str) -> Dict[str, Dict[str, str]]:
    # get valid files from dir
    files = os.listdir(dir_path)

    # only take dirs with _stats_
    valid_files = []
    for _file in files:
        if '_stats_' not in _file:
            continue
        valid_files.append(_file)

    # collect data across multiple files
    data = {}

    for _file in valid_files:
        path = os.path.join(dir_path, valid_files)

        with open(path, 'r') as f:
            topic = f.readline().split('Topic:')[1].strip()
            f.readline()
            stats = f.readline()

        topic_info = {}
        for stat in stats.split(','):
            key, val = stat.split(':')
            topic_info[key.strip()] = val.strip()

        data[topic] = topic_info
        
    return data

def main() -> None:
    pass

if __name__ == '__main__':
    main()
