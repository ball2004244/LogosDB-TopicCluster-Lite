import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This file aggregates the raw_call benchmark records from a dir on a specific subject and generates various analytical stats & figures
'''

def get_subjects(base_dir: str) -> list:
    '''
    Get the list of subjects by listing all folders in the base_dir
    '''
    subjects = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    return subjects

def process_multi_calls(base_dir: str,  nat_science: list, soc_science: list, humanities: list) -> pd.DataFrame:
    '''
    Prerequisite: 
    + Run Benchmark multi calls on a specific subject first
    + Run Measure multi calls to generate stats files

    Then run this function to aggregate stats records from multiple files of the same subject and generate stats
    '''
    subjects = nat_science + soc_science + humanities
    data = []

    for subject in subjects:
        dir_path = os.path.join(base_dir, subject)
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

def agg_visualize(df: pd.DataFrame, save_path: str = 'analysis/figures') -> bool:
    '''
    This function will aggregate all subjects and generate a single figure
    '''
    print('Starting to aggregate visualizations...')

    # Get unique subjects
    subjects = df['topic'].unique()
    plt.figure(figsize=(20, 15))

    # Dictionary to keep track of legend entries
    legend_entries = {'nat_science': False, 'soc_science': False, 'humanities': False}

    for subject in subjects:
        # Modify the subject to be more readable
        col_name = subject.replace('_', ' ')

        # Calculate the median and percentiles of accuracy
        median_accuracy = df[df['topic'] == subject]['median_accuracy'].values[0]
        percentile_25 = df[df['topic'] == subject]['percentile_25'].values[0]
        percentile_75 = df[df['topic'] == subject]['percentile_75'].values[0]

        # Determine the category
        category = df[df["topic"] == subject]["category"].values[0]

        # Plot the median accuracy as a point with a vertical error line
        if category == 'nat_science':
            plt.errorbar(col_name, median_accuracy, yerr=[[median_accuracy - percentile_25], [percentile_75 - median_accuracy]],
                        fmt='o', color='blue', ecolor='black', capsize=12, markersize=10, 
                        label='Natural Science' if not legend_entries['nat_science'] else "")
            legend_entries['nat_science'] = True
        elif category == 'soc_science':
            plt.errorbar(col_name, median_accuracy, yerr=[[median_accuracy - percentile_25], [percentile_75 - median_accuracy]],
                        fmt='o', color='green', ecolor='black', capsize=12, markersize=10, 
                        label='Social Science' if not legend_entries['soc_science'] else "")
            legend_entries['soc_science'] = True
        elif category == 'humanities':
            plt.errorbar(col_name, median_accuracy, yerr=[[median_accuracy - percentile_25], [percentile_75 - median_accuracy]],
                        fmt='o', color='red', ecolor='black', capsize=12, markersize=10, 
                        label='Humanities' if not legend_entries['humanities'] else "")
            legend_entries['humanities'] = True
    plt.title('Accuracy for Each Subject')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=30)
    plt.legend()

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
    aggregated_stats = df.groupby(['topic', 'category']).agg(
        min_accuracy=('Accuracy', 'min'),
        max_accuracy=('Accuracy', 'max'),
        median_accuracy=('Accuracy', 'median'),
        percentile_25=('Accuracy', lambda x: x.quantile(0.25)),
        percentile_75=('Accuracy', lambda x: x.quantile(0.75))
    ).reset_index()
    
    return aggregated_stats

def categorize_subjects(df: pd.DataFrame, nat_science: list, soc_science: list, humanities: list) -> pd.DataFrame:
    '''
    Add a category column to the DataFrame based on the provided lists of natural science and social science subjects
    '''
    df['category'] = df['topic'].apply(lambda x: 'nat_science' if x in nat_science else 'soc_science' if x in soc_science else 'humanities')
    return df

def main() -> None:
    res_dir = 'results/auxi_logos_extract/'

    # Get all subjects in the res_dir
    subjects = get_subjects(res_dir)
    print(f'Found subjects: {subjects}')

    # Define the lists of natural science and social science subjects
    nat_science = [
        'abstract_algebra',
        'college_physics',
        'electrical_engineering',
        'high_school_biology',
        'machine_learning',
        'high_school_chemistry',
    ]
    soc_science = [
        'high_school_geography',
        'sociology',
        'high_school_macroeconomics',
        'professional_psychology',
        'human_sexuality',
        'public_relations',
    ]
    humanities = [
        'high_school_world_history',
        'logical_fallacies',
        'world_religions',
        'philosophy',
        'moral_disputes',
        'business_ethics',
    ]

    print(f'Aggregating stats across {len(nat_science) + len(soc_science) + len(humanities)} subjects...')
    start = time.perf_counter()
    concat_df = pd.DataFrame()
    
    df = process_multi_calls(res_dir, nat_science, soc_science, humanities)
    concat_df = pd.concat([concat_df, df])
    print("concat_df\n",concat_df)

    # Categorize subjects
    categorized_df = categorize_subjects(concat_df, nat_science, soc_science, humanities)
    print(categorized_df)

    # Calculate aggregated statistics
    aggregated_stats_df = calculate_aggregated_stats(categorized_df)
    
    # Define the desired order for the categories
    category_order = ['nat_science', 'soc_science', 'humanities']

    # Convert the 'category' column to a categorical type with the specified order
    aggregated_stats_df['category'] = pd.Categorical(aggregated_stats_df['category'], categories=category_order, ordered=True)

    # Sort the DataFrame by the 'category' column
    aggregated_stats_df = aggregated_stats_df.sort_values(by='category')

    print(aggregated_stats_df)

    agg_visualize(aggregated_stats_df, save_path='test/auxi_logos_extract/figures')

    print(f'Aggregation done in {time.perf_counter() - start} seconds.')  

if __name__ == '__main__':
    main()