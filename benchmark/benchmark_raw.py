
import os
import time
from constants import SUBJECT, PROMPT, SUFFIX_PROMPT, OLLAMA_MODEL
from ollama import raw_call
from datasets import load_dataset
import pandas as pd
import string

'''
This file is for benchmarking pure LLama 3.1 8B logical thinking on MMLU dataset.

MUST Run LLama 3.1 8B First with the following command:
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

'''


def benchmark_raw(df: pd.DataFrame, res_dir: str = 'results', res_file: str = 'llama_raw.txt') -> None:
    print('Starting benchmarking on raw SLM...')
    start = time.perf_counter()
    res_path = os.path.join(res_dir, res_file)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    topic = df['subject'][0]
    with open(res_path, 'w') as f:
        f.write(f'Topic: {topic}\n')

    for i, row in df.iterrows():
        print(f'Processing row {i}/{len(df)}...')
        question = row['question']
        choices = row['choices']

        # Generate labels dynamically based on the length of lst
        labels = list(string.ascii_uppercase[:len(choices)])
        formatted_list = [f"{label}. {item}" for label,
                          item in zip(labels, choices)]
        choices = "; ".join(formatted_list)

        prompt = PROMPT % (SUBJECT)
        end_suffix_prompt = SUFFIX_PROMPT % (question, choices)

        prompt = f'{prompt}\n{end_suffix_prompt}'

        raw_res = raw_call(prompt, model=OLLAMA_MODEL)

        # extract the final answer from the raw response
        res = raw_res.split('Final answer: ')[-1].strip()
        with open(res_path, 'a') as f:
            f.write(f'{res}\n')
            # f.write(f'Question: {question}\n')
            # f.write(f'Choices: {choices}\n')
            # f.write(f'Answer: {res}\n')
            # f.write('--------------------------\n')

    print(f'Benchmarking done in {time.perf_counter() - start} seconds.')


def auto_benchmark(df: pd.DataFrame, num_calls: int = 1) -> bool:
    try:
        print(f'STARTING AUTO BENCHMARK WITH {num_calls} CALLS...')
        res_dir = f'results/raw_multi_calls_{SUBJECT}'
        res_file = 'llama_raw_%d.txt'
        for i in range(num_calls):
            print(f'Auto benchmarking call {i+1}/{num_calls}...')
            benchmark_raw(df, res_dir=res_dir, res_file=(res_file % i))
            print(f'Finished benchmarking call {i+1}/{num_calls}...')

        print(f'AUTO BENCHMARK DONE!')
        return True

    except Exception as e:
        print(f'Error on auto_benchmark: {e}')
        return False


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Topic: {df["subject"][0]}')
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    # benchmark_raw(df)
    num_calls = 100
    auto_benchmark(df, num_calls=num_calls)
