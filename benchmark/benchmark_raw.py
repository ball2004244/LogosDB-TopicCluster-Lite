
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


def benchmark_raw(df: pd.DataFrame) -> None:
    print('Starting benchmarking on raw SLM...')
    start = time.perf_counter()
    res_dir = 'results'
    res_file = 'llama_raw.txt'
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
        formatted_list = [f"{label}. {item}" for label, item in zip(labels, choices)]
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


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Topic: {df["subject"][0]}')
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_raw(df)
