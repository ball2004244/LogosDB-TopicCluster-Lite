from typing import List
from datasets import load_dataset
from constants import SUBJECT
from benchmark_raw import benchmark_raw
from benchmark_auxi_train import benchmark_auxi_train
from measure import measure_slm_results
import os
import sys
import time
import pandas as pd

# Add the absolute path of parent dir to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import log, LogType


'''
A comprehensive testing file for benchmarking and measuring the SLM performance multiple times.

It do multi-benchmarking LLama 3.1 8B logical thinking on MMLU dataset. It will run on multiple subjects of MMLU instead of just one.

Every subject is run at least n times.
'''


def auto_benchmark(df: pd.DataFrame, benchmark_func: callable = benchmark_raw, subject: str = SUBJECT, num_calls: int = 1, res_dir: str = 'results') -> bool:
    '''
    Benchmark a given function with multiple calls.
    '''
    try:
        log(
            f'STARTING AUTO BENCHMARK ON {subject} WITH {num_calls} CALLS...', LogType.INFO)
        res_file = 'llama_raw_%d.txt'

        os.makedirs(res_dir, exist_ok=True)

        for i in range(num_calls):
            start = time.perf_counter()
            log(f'Auto benchmarking call {i+1}/{num_calls}...', LogType.INFO)
            benchmark_func(df, res_dir=res_dir, res_file=(
                res_file % i), subject=subject)

            elapsed = time.perf_counter() - start
            log(f'Finished benchmarking call {i+1}/{num_calls}...',
                LogType.SUCCESS)
            log(f'Elapsed time: {elapsed:.4f} seconds (~ {elapsed/3600:.2f} hours).', LogType.INFO)

        log(f'AUTO BENCHMARK DONE FOR {subject}!', LogType.SUCCESS)
        return True

    except Exception as e:
        log(f'Error on auto_benchmark: {e}', LogType.ERROR)
        return False


def multi_benchmark(subjects: List[str], benchmark_func: callable = benchmark_raw, num_calls: int = 100, res_dir: str = 'results') -> None:
    '''
    Auto benchmark multiple subjects of MMLU dataset.
    '''
    log('STARTING MULTI BENCHMARK...', LogType.INFO)
    log('CONFIGS:', LogType.INFO)
    log(f'Benchmark function: {benchmark_func.__name__}', LogType.INFO)
    log(f'Number of calls: {num_calls}', LogType.INFO)
    log(f'Subjects count: {len(subjects)}', LogType.INFO)
    log(f'All Subjects: {subjects}', LogType.INFO)

    start = time.perf_counter()
    for i, sub in enumerate(subjects):
        start = time.perf_counter()
        log(
            f'MULTI BENCHMARK - Processing subject {i+1}/{len(subjects)}: {sub}...', LogType.INFO)
        ds = load_dataset("cais/mmlu", sub)
        df = pd.DataFrame(ds['test'])

        auto_benchmark(df, benchmark_func=benchmark_func,
                       subject=sub, num_calls=num_calls, res_dir=res_dir)

        elapsed = time.perf_counter() - start
        log(
            f'MULTI BENCHMARK - Finished processing subject {i+1}/{len(subjects)}: {sub}...', LogType.SUCCESS)
        log(
            f'MULTI BENCHMARK - Elapsed time: {elapsed:.4f} seconds (~ {elapsed/3600:.2f} hours).', LogType.INFO)

    elapsed = time.perf_counter() - start
    log('MULTI BENCHMARK DONE!', LogType.SUCCESS)
    log(f'TOTAL MULTI BENCHMARK ELAPSED TIME: {elapsed:.4f} seconds (~ {elapsed/3600:.2f} hours).', LogType.INFO)


def auto_measure(df: pd.DataFrame, measure_func: callable = measure_slm_results, subject: str = SUBJECT, res_dir: str = 'results', res_file: str = 'llama_raw_0.txt') -> bool:
    '''
    Generate benchmark stats for a given function with multiple calls.
    '''
    try:
        log(f'STARTING AUTO MEASURE...', LogType.INFO)
        res_dir = os.path.join(res_dir, subject)
        res_file = 'llama_raw_%d.txt'

        if not os.path.exists(res_dir):
            log(f'Error: No benchmark results found for {subject}!', LogType.ERROR)
            return False

        # get all files in a directory
        num_files = os.listdir(res_dir)

        valid_num_calls = [f for f in num_files if '_stats.txt' not in f]

        for i in range(len(valid_num_calls)):
            log(f'Measuring file {i+1}/{len(valid_num_calls)}...', LogType.INFO)
            measure_func(df, res_dir=res_dir, res_file=(res_file % i))
            log(f'Finished measuring file {i+1}/{len(valid_num_calls)}...',
                LogType.SUCCESS)

        log(f'AUTO MEASURE DONE!', LogType.SUCCESS)
        return True
    except Exception as e:
        log(f'Error on auto_measure: {e}', LogType.ERROR)
        return False


def multi_measure(subjects: List[str], measure_func: callable = measure_slm_results, res_dir: str = 'results') -> None:
    '''
    Auto measure & create a report for multiple subjects of MMLU dataset.
    '''
    start = time.perf_counter()
    log(
        f'MULTI MEASURE - Gathering information from {len(subjects)} subjects...', LogType.INFO)
    for sub in subjects:
        ds = load_dataset("cais/mmlu", sub)
        df = pd.DataFrame(ds['test'])
        auto_measure(df, subject=sub, measure_func=measure_func, res_dir=res_dir)
    elapsed = time.perf_counter() - start
    log(
        f'MULTI MEASURE - Finished gathering information from {len(subjects)} subjects...', LogType.SUCCESS)
    log(f'MULTI MEASURE - Elapsed time: {elapsed:.4f} seconds (~ {elapsed/3600:.2f} hours).', LogType.INFO)


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    # subjects = [
    #     'astronomy',
    #     'high_school_chemistry',
    #     'college_chemistry',
    #     'high_school_macroeconomics',
    #     'high_school_mathematics',
    #     'college_mathematics',
    #     'elementary_mathematics',
    #     'high_school_microeconomics',
    #     'high_school_physics',
    #     'college_physics',
    #     'conceptual_physics',
    #     'high_school_psychology',
    #     'professional_psychology',
    #     'philosophy',
    # ]
    subjects = [
        'astronomy'
    ]
    num_calls = 5

    #! do multi benchmark on LLama call
    # multi_benchmark(subjects, benchmark_raw, num_calls, res_dir='results')
    multi_benchmark(subjects, benchmark_auxi_train, num_calls, res_dir='results/auxi_train')
    multi_measure(subjects, measure_slm_results)
