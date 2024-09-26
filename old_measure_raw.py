from constants import SUBJECT, ANSWER_MAP
from datasets import load_dataset
import os
import pandas as pd

'''
This file will compared generated answers from SLM with given answers in the dataset to measure the accuracy of the model.
'''


def measure_raw(df: pd.DataFrame, res_dir: str = 'results', res_file: str = 'llama_raw.txt') -> None:
    print('Starting measuring on raw SLM...')

    # read the result file
    res_path = os.path.join(res_dir, res_file)
    with open(res_path, 'r') as f:
        topic_row = f.readline()
        res = f.readlines()

    if len(res) != len(df):
        print(
            f'Error: The number of answers in {res_file} is not equal to the number of questions in the dataset.')
        return

    # calculate the accuracy
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

    # save stats to file
    stats_file = f'{res_file.split(".")[0]}_stats.txt'
    stats_path = os.path.join(res_dir, stats_file)
    with open(stats_path, 'w') as f:
        f.write(f'{topic_row}\n')
        f.write(
            f'Correct: {correct}, Wrong: {wrong}, Accuracy: {correct/len(df)}')
    print(f'Save result to {stats_path}')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    df = pd.DataFrame(ds['test'])
    measure_raw(df)
