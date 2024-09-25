import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.constants import SUBJECT

'''
This file aggregate the raw_call benchmark records from a dir on specific subject and generate various analytical stats & figures
'''


def process_multi_calls(dir_path: str) -> pd.DataFrame:
    '''
    Prerequiste: 
    + Run Benchmark multi calls on a specific subject first
    + Run Measure multi calls to generate stats files

    Then run this function to aggergate stats records from multiple file of the same subject and generate stats
    '''
    # get valid files from dir
    files = os.listdir(dir_path)

    # only take files with _stats_ in name
    valid_files = [f for f in files if '_stats' in f]

    # collect data across multiple files
    data = []

    for _file in valid_files:
        path = os.path.join(dir_path, _file)

        with open(path, 'r') as f:
            topic = f.readline().split('Topic:')[1].strip()
            f.readline()
            stats = f.readline()

        topic_info = {'topic': topic}
        for stat in stats.split(','):
            key, val = stat.split(':')
            topic_info[key.strip()] = val.strip()

        data.append(topic_info)

    df = pd.DataFrame(data)
    # Add an index column starting from 1
    df.index = df.index + 1
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Index'}, inplace=True)

    return df


def visualize(df: pd.DataFrame, save_path: str = 'analysis/figures') -> bool:
    # Convert Accuracy column to numeric
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')

    # Plot the Accuracy column as a box plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Accuracy')
    plt.title('Accuracy Box Plot')
    plt.ylabel('Accuracy')

    # save to file
    plt.savefig(os.path.join(save_path, f'{SUBJECT}_accuracy.png'))

    print(
        f'Save figure to {os.path.join(save_path, f"{SUBJECT}_accuracy.png")}')
    return True


def main() -> None:
    res_dir = 'results'
    sub_res = f'raw_multi_calls_{SUBJECT}'
    dir_path = os.path.join(res_dir, sub_res)

    df = process_multi_calls(dir_path)

    visualize(df)


if __name__ == '__main__':
    main()
