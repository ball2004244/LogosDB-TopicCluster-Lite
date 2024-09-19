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

    res_path = os.path.join(res_dir, res_file)
    with open(res_path, 'r') as f:
        topic_row = f.readline()
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
        f.write(f'{topic_row}\n')
        f.write(
            f'Correct: {correct}, Wrong: {wrong}, Accuracy: {correct/len(df)}')
    print(f'Save result to {stats_path}')

def measure_rag(df: pd.DataFrame) -> None:
    print('Starting measuring on RAG SLM...')
    res_dir = 'results'
    res_file = 'llama_logos.txt'

    res_path = os.path.join(res_dir, res_file)
    topic_row = ''
    res = []
    with open(res_path, 'r') as f:
        # Skip the first row (topic)
        topic_row = f.readline()
        # Each res is actually a para until meeting delimiter: '---------------------------------------'
        raw_file = f.read()

        # Split by 'Final Choice'
        splitted_file = raw_file.split('Final Choice:')[1:]
        answer = 'F' # default to F
        for raw_ans in splitted_file:
            # take only the first character of each answer
            answer = raw_ans.strip()[0].upper()
            res.append(answer)

    debug_dir = 'debug'
    debug_file = f'{res_file.split(".")[0]}_measure_debug.txt'
    debug_path = os.path.join(debug_dir, debug_file)
    with open(debug_path, 'w') as f:
        f.write(f'{topic_row}\n')
        for i, r in enumerate(res):
            f.write(f'{i+1}. {r}\n')

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
        f.write(f'{topic_row}\n')
        f.write(
            f'Correct: {correct}, Wrong: {wrong}, Accuracy: {correct/len(df):.3f}')
    print(f'Save result to {stats_path}')

if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    df = pd.DataFrame(ds['test'])
    # measure_raw(df)
    measure_rag(df)
