from typing import List
from datasets import load_dataset
from constants import SUBJECT
from benchmark_raw import benchmark_raw
from measure_raw import measure_raw
import os
import pandas as pd

'''
This file is for multi-benchmarking LLama 3.1 8B logical thinking on MMLU dataset.

It will run on multiple subjects of MMLU instead of just one.

Every subject is run at least n times.
'''


def auto_benchmark(df: pd.DataFrame, benchmark_func: callable = benchmark_raw,
                   subject: str = SUBJECT, num_calls: int = 1) -> bool:
    '''
    Benchmark a given function with multiple calls.
    '''
    try:
        print(f'STARTING AUTO BENCHMARK WITH {num_calls} CALLS...')
        res_dir = f'results/raw_multi_calls_{subject}'
        res_file = 'llama_raw_%d.txt'
        for i in range(num_calls):
            print(f'Auto benchmarking call {i+1}/{num_calls}...')
            benchmark_func(df, res_dir=res_dir, res_file=(res_file % i))
            print(f'Finished benchmarking call {i+1}/{num_calls}...')

        print(f'AUTO BENCHMARK DONE!')
        return True

    except Exception as e:
        print(f'Error on auto_benchmark: {e}')
        return False


def multi_benchmark(subject: List[str], benchmark_func: callable = benchmark_raw, num_calls: int = 100) -> None:
    '''
    Auto benchmark multiple subjects of MMLU dataset.
    '''
    for i, sub in enumerate(subject):
        print(f'Processing subject {i+1}/{len(subject)}: {sub}...')
        ds = load_dataset("cais/mmlu", sub)
        df = pd.DataFrame(ds['test'])

        auto_benchmark(df, benchmark_func=benchmark_func,
                       subject=sub, num_calls=num_calls)


def auto_measure(df: pd.DataFrame, measure_func: callable = measure_raw, subject: str = SUBJECT) -> bool:
    '''
    Generate benchmark stats for a given function with multiple calls.
    '''
    try:
        print(f'STARTING AUTO MEASURE...')
        res_dir = os.path.join('results', f'raw_multi_calls_{subject}')
        res_file = 'llama_raw_%d.txt'

        # get all files in a directory
        num_files = os.listdir(res_dir)

        valid_num_calls = [f for f in num_files if '_stats_' not in f]

        for i in range(len(valid_num_calls)):
            print(f'Measuring file {i+1}/{len(valid_num_calls)}...')
            measure_func(df, res_dir=res_dir, res_file=(res_file % i))
            print(f'Finished measuring file {i+1}/{len(valid_num_calls)}...')

        print(f'AUTO MEASURE DONE!')
        return True
    except Exception as e:
        print(f'Error on auto_measure_raw: {e}')
        return False


def multi_measure(subject: List[str], measure_func: callable = measure_raw) -> None:
    '''
    Auto measure & create a report for multiple subjects of MMLU dataset.
    '''
    for sub in subject:
        ds = load_dataset("cais/mmlu", sub)
        df = pd.DataFrame(ds['test'])
        auto_measure(df, subject=sub, measure_func=measure_func)


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    subjects = [
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
    ]
    num_calls = 20

    #! do multi benchmark on raw LLama call
    multi_benchmark(subjects, benchmark_raw, num_calls)
    multi_measure(subjects, measure_raw)

    total = time.perf_counter() - start
    total_min = total // 60
    print(
        f'Finished processing {len(subjects)} subjects in {total:.4f} seconds (~ {total_min:.4f} minutes), with {num_calls} calls each.')
