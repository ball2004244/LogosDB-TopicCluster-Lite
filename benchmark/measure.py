from constants import SUBJECT, ANSWER_MAP
from datasets import load_dataset
import os
import pandas as pd

'''
This file will compared generated answers from SLM with given answers in the dataset to measure the accuracy of the model.
'''


def measure_raw(df: pd.DataFrame) -> None:
    print('Starting measuring on raw SLM...')
    res_dir = 'results'
    res_file = 'llama_raw.txt' # For raw SLM
    #res_file = 'llama_logos.txt' # For SLM + LogosDB

    res_path = os.path.join(res_dir, res_file)
    with open(res_path, 'r') as f:
        # Skip the first row (topic)
        f.readline()
        res = f.readlines()
    correct = 0
    wrong = 0
    for i, row in df.iterrows():
        print(f'Processing row {i+1}/{len(df)}...')
        # also take only the first character
        raw_answer = res[i].strip()[0].upper()
        answer = row['answer']

        if raw_answer not in ANSWER_MAP:
            wrong += 1
        elif ANSWER_MAP[raw_answer] == answer:
            correct += 1
        else:
            wrong += 1

    # save the result to a file
    stats_file = f'{res_file.split(".")[0]}_stats.txt'
    stats_path = os.path.join(res_dir, stats_file)
    with open(stats_path, 'w') as f:
        f.write(
            f'Correct: {correct}, Wrong: {wrong}, Accuracy: {correct/len(df)}')
    print(f'Save result to {stats_path}')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    df = pd.DataFrame(ds['test'])
    measure_raw(df)
