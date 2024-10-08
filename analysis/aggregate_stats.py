import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    valid_files = [f for f in files if '_stats.txt' in f]

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


def visualize(df: pd.DataFrame, subject: str, save_path: str = 'analysis/figures') -> bool:
    '''
    Visualize for each subject
    '''
    # Convert Accuracy column to numeric
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')

    # Calculate the median and range of accuracy
    median_accuracy = df['Accuracy'].median()
    min_accuracy = df['Accuracy'].min()
    max_accuracy = df['Accuracy'].max()

    # Plot the vertical error line to denote the range of accuracy
    plt.figure(figsize=(10, 6))
    plt.errorbar(subject, median_accuracy, yerr=[[median_accuracy - min_accuracy], [
                 max_accuracy - median_accuracy]], fmt='o', color='black', ecolor='red', capsize=12, markersize=10)

    plt.title('Accuracy for single Subject')
    plt.ylabel('Accuracy')

    # Save to file
    plt.savefig(os.path.join(save_path, f'{subject}_accuracy.png'))

    print(
        f'Save figure to {os.path.join(save_path, f"{subject}_accuracy.png")}')
    return True


def agg_visualize(df: pd.DataFrame, save_path: str = 'analysis/figures') -> bool:
    '''
    This function will aggregate all subjects and generate a single figure
    '''
    print('Starting to aggregate visualizations...')
    # Convert Accuracy column to numeric
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')

    # Check for NaN values in Accuracy column
    if df['Accuracy'].isnull().any():
        print("Warning: There are NaN values in the 'Accuracy' column after conversion.")

    plt.figure(figsize=(16, 10))
    plt.ylim(0.2, 0.8)
    # Get unique subjects
    subjects = df['topic'].unique()
    for subject in subjects:
        # Filter data for the current subject
        subject_data = df[df['topic'] == subject]

        # modify the subject to be more readable
        col_name = subject.replace('_', ' ')

        # Calculate the median and percentiles of accuracy
        median_accuracy = subject_data['Accuracy'].median()
        percentile_25 = subject_data['Accuracy'].quantile(0.25)
        percentile_75 = subject_data['Accuracy'].quantile(0.75)

        # Plot the median accuracy as a red point with a vertical error line
        plt.errorbar(col_name, median_accuracy, yerr=[[median_accuracy - percentile_25], [percentile_75 - median_accuracy]],
                     fmt='o', color='red', ecolor='black', capsize=12, markersize=10, label=f'{col_name} Median with IQR')

    plt.title('Accuracy for Each Subject')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    # Save to file
    plt.savefig(os.path.join(save_path, 'aggr_accuracy.png'))

    print(
        f'Finished aggregate visualizations. Save figure to {os.path.join(save_path, "aggr_accuracy.png")}')
    return True


def main() -> None:
    res_dir = 'results/auxi_logos_extract'
    sub_res = '%s'
    save_path ='analysis/figures/auxi_logos_extract/humanities'

    # subjects = [
    #     'abstract_algebra',
    #     'college_physics',
    #     'electrical_engineering',
    #     'high_school_biology',
    #     'machine_learning',
    #     'high_school_chemistry',
    # ]


    """
    subjects = [
        'high_school_geography',
        'sociology',
        'high_school_macroeconomics',
        'professional_psychology',
        'human_sexuality',
        'public_relations',
    ]
    """
    subjects = [
        'high_school_world_history',
        'logical_fallacies',
        'world_religions',
        'philosophy',
        'professional_law',
        'moral_scenarios',
    ]

    print(f'Aggregating stats across {len(subjects)} subjects...')
    start = time.perf_counter()
    concat_df = pd.DataFrame()
    for sub in subjects:
        dir_path = os.path.join(res_dir, sub_res % sub)
        df = process_multi_calls(dir_path)
        visualize(df, sub, save_path)
        concat_df = pd.concat([concat_df, df])

    agg_visualize(concat_df, save_path)

    print(f'Aggregation done in {time.perf_counter() - start} seconds.')


if __name__ == '__main__':
    main()
