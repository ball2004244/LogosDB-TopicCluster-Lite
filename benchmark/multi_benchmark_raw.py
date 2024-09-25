
from typing import List
from datasets import load_dataset
from benchmark_raw import auto_benchmark
from measure_raw import auto_measure_raw
import pandas as pd

'''
This file is for benchmarking pure LLama 3.1 8B logical thinking on MMLU dataset.

it will run on multiple subjects of MMLU instead of just one.

Every subject is run at least 100 times.
'''

def multi_benchmark(subject: List[str], num_calls: int=100) -> None:
    '''
    Auto benchmark multiple subjects of MMLU dataset.
    '''
    for i, sub in enumerate(subject):
        print(f'Processing subject {i+1}/{len(subject)}: {sub}...')
        ds = load_dataset("cais/mmlu", sub)
        df = pd.DataFrame(ds['test'])

        auto_benchmark(df, subject=sub,num_calls=num_calls)

def multi_measure(subject: List[str]) -> None:
    '''
    Auto measure & create a report for multiple subjects of MMLU dataset.
    '''
    for sub in subject:
        ds = load_dataset("cais/mmlu", sub)
        df = pd.DataFrame(ds['test'])
        auto_measure_raw(df, subject=sub)

if __name__ == '__main__':
    import time
    start = time.perf_counter()
    subjects = [
        'college_biology',
        'college_computer_science',
        'college_chemistry',
        'college_medicine'
    ]
    num_calls = 100

    multi_benchmark(subjects, num_calls)
    multi_measure(subjects)

    total = time.perf_counter() - start
    total_min = total // 60
    print(f'Finished processing {len(subjects)} subjects in {total} seconds (~ {total_min} minutes), with {num_calls} calls each.')