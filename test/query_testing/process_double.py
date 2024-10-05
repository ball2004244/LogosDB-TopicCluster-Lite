import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This file aggregates the raw_call benchmark records from a dir on a specific subject and generates various analytical stats & figures
'''

def process_multi_calls(base_dir: str, folders: list, subjects: list) -> pd.DataFrame:
    '''
    Prerequisite: 
    + Run Benchmark multi calls on a specific subject first
    + Run Measure multi calls to generate stats files

    Then run this function to aggregate stats records from multiple files of the same subject and generate stats
    '''
    data = []

    for folder in folders:
        for subject in subjects:
            dir_path = os.path.join(base_dir, folder, subject)
            # get valid files from dir
            files = os.listdir(dir_path)

            # only take files with _stats_ in name
            valid_files = [f for f in files if '_stats.txt' in f]

            # collect data across multiple files
            for _file in valid_files:
                path = os.path.join(dir_path, _file)

                with open(path, 'r') as f:
                    topic = f.readline().split('Topic:')[1].strip()
                    f.readline()
                    stats = f.readline()

                topic_info = {'topic': topic, 'folder': folder}
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

def visualize(df: pd.DataFrame, topic: str, save_path: str = 'analysis/figures') -> bool:
    '''
    Visualize for each topic
    '''
    df_sub = df[df['topic'] == topic]
    print(df_sub)
    folders = df_sub['folder'].unique()

    # Plot the vertical error line to denote the range of accuracy
    plt.figure(figsize=(10, 6))
    for folder in folders:
        folder_data = df_sub[df_sub['folder'] == folder]
        median_accuracy = folder_data['median_accuracy'].values[0]
        min_accuracy = folder_data['min_accuracy'].values[0]
        max_accuracy = folder_data['max_accuracy'].values[0]
        plt.errorbar(folder, median_accuracy, yerr=[[median_accuracy - min_accuracy], [
                    max_accuracy - median_accuracy]], fmt='o', color='black', ecolor='red', capsize=12, markersize=10)

    plt.title(f'Accuracy for Topic: {topic}')
    plt.ylabel('Accuracy')

    # Save to file
    plt.savefig(os.path.join(save_path, f'{topic}_accuracy.png'))

    print(f'Save figure to {os.path.join(save_path, f"{topic}_accuracy.png")}')
    return True

def agg_visualize(df: pd.DataFrame, subjects: list, save_path: str = 'analysis/figures') -> bool:
    '''
    This function will aggregate all subjects and generate a single figure
    '''
    print('Starting to aggregate visualizations...')

    # Split the DataFrame into two based on folder
    df_smart = df[df['folder'] == 'smart']
    df_improved = df[df['folder'] == 'improved']

    print(df_smart)
    print(df_improved)
    
    fig, ax = plt.subplots(figsize=(16, 10))

    x = np.arange(len(subjects))

    # Plot error bars for df_smart
    ax.errorbar(x - 0.1, df_smart['median_accuracy'], 
                yerr=[df_smart['median_accuracy'] - df_smart['min_accuracy'], 
                      df_smart['max_accuracy'] - df_smart['median_accuracy']], 
                fmt='o', color='orange', ecolor='black', label='Smart', capsize=5)

    # Plot error bars for df_improved
    ax.errorbar(x + 0.1, df_improved['median_accuracy'], 
                yerr=[df_improved['median_accuracy'] - df_improved['min_accuracy'], 
                      df_improved['max_accuracy'] - df_improved['median_accuracy']], 
                fmt='o', color='red', ecolor='black', label='Improved', capsize=5)
   
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by subject and folder')
    ax.set_xticks(x)
    ax.set_xticklabels(df_smart['topic'])
    ax.legend()

    # Save to file
    plt.savefig(os.path.join(save_path, 'aggr_accuracy.png'))

    print(f'Finished aggregate visualizations. Save figure to {os.path.join(save_path, "aggr_accuracy.png")}')
    return True

def calculate_aggregated_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate min, max, and median accuracy by topic and folder, and add to another DataFrame
    '''
    # Convert Accuracy column to numeric
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')

    # Group by topic and folder and calculate aggregated statistics
    aggregated_stats = df.groupby(['topic', 'folder']).agg(
        min_accuracy=('Accuracy', 'min'),
        max_accuracy=('Accuracy', 'max'),
        median_accuracy=('Accuracy', 'median'),
        percentile_25=('Accuracy', lambda x: x.quantile(0.25)),
        percentile_75=('Accuracy', lambda x: x.quantile(0.75))
    ).reset_index()
    
    return aggregated_stats

def main() -> None:
    res_dir = 'results/query_testing/'
    folders = [
        'improved',
        'smart'
    ]
    subjects = [
        'college_computer_science',
        'college_biology',
        'college_chemistry',
        'college_physics',
        'college_mathematics',
        'college_medicine'
    ]

    print(f'Aggregating stats across {len(subjects)} subjects...')
    start = time.perf_counter()
    concat_df = pd.DataFrame()
    
    df = process_multi_calls(res_dir, folders, subjects)
    concat_df = pd.concat([concat_df, df])
    print(concat_df)

    # Calculate aggregated statistics
    aggregated_stats_df = calculate_aggregated_stats(concat_df)
    print(aggregated_stats_df)
    
    for sub in subjects:
        visualize(aggregated_stats_df, sub, save_path='test/query_testing/figures')

    agg_visualize(aggregated_stats_df, subjects, save_path='test/query_testing/figures')

    print(f'Aggregation done in {time.perf_counter() - start} seconds.')  

if __name__ == '__main__':
    main()