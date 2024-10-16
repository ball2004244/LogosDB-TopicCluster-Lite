from aggregate_stats import process_multi_calls, agg_visualize
import pandas as pd
import os

def aggregate_df(res_dir: str = 'results') -> pd.DataFrame:
    nat_sci = [
        'abstract_algebra',
        'college_physics',
        'electrical_engineering',
        'high_school_biology',
        'machine_learning',
        'high_school_chemistry',
    ]

    soc_sci = [
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
        'business_ethics',
        'moral_disputes',
    ]
    
    subjects = [*nat_sci, *soc_sci, *humanities]

    concat_df = pd.DataFrame()
    for sub in subjects:
        dir_path = os.path.join(res_dir, sub)
        df = process_multi_calls(dir_path)
        concat_df = pd.concat([concat_df, df])
        
    return concat_df

def calc_acc_per_topic(df: pd.DataFrame) -> pd.DataFrame:
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')

    return df.groupby('topic')['Accuracy'].mean().reset_index()

def calc_acc_all_topics(df: pd.DataFrame) -> float:
    return df['Accuracy'].mean()

if __name__ == '__main__':
    res_dir_prefix = 'results'
    save_path_prefix = 'analysis/figures'
    
    # process_dir = 'auxi_logos_abstract' # For measuring AuxiLogos Abstract
    # process_dir = 'auxi_logos_extract' # For measuring AuxiLogos
    process_dir = 'auxi_db' # For measuring accuracy of AuxiDB
    # process_dir = 'raw' # For visualizing raw call without RAG

    
    res_dir = os.path.join(res_dir_prefix, process_dir)
    save_path = os.path.join(save_path_prefix, process_dir)
    concat_df = aggregate_df(res_dir)
    
    os.makedirs(save_path, exist_ok=True)
    agg_visualize(concat_df, save_path)
    
    avg_acc = calc_acc_per_topic(concat_df)
    print(avg_acc)
    
    print(f'Average accuracy across all topics: {calc_acc_all_topics(concat_df):.3f}')
