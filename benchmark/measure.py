from constants import SUBJECT, ANSWER_MAP
from datasets import load_dataset
import os
import pandas as pd

'''
This file will compared generated answers from SLM with given answers in the dataset to measure the accuracy of the model.
'''


def measure_slm_results(df: pd.DataFrame, res_dir: str = 'results', res_file: str = 'llama_auxi_db.txt') -> None:
    print('Starting measuring on RAG SLM...')
    res_path = os.path.join(res_dir, res_file)
    topic_row = ''
    res = []
    with open(res_path, 'r') as f:
        # Skip the first row (topic row)
        topic_row = f.readline()
        raw_file = f.read()

        # first split by '---' and drop the last empty string
        splitted_file = raw_file.split(
            '---------------------------------------')[:-1]

        answer = 'F'  # default ans
        for ans in splitted_file:
            # check if final choice present, if not append 'F'

            if 'Final Choice:' not in ans:
                res.append(answer)
                continue

            # Split by 'Final Choice' and take the second part
            splitted_ans = ans.split('Final Choice:')[1]

            if not splitted_ans.strip():
                res.append(answer)
                continue

            # take only the first character of each answer
            final_answer = splitted_ans.strip()[0].upper()
            res.append(final_answer)

    #! Uncomment this block to debug the result
    # debug_dir = 'debug'
    # debug_file = f'{res_file.split(".")[0]}_measure_debug.txt'
    # debug_path = os.path.join(debug_dir, debug_file)
    # with open(debug_path, 'w') as f:
    #     f.write(f'{topic_row}\n')
    #     for i, r in enumerate(res):
    #         f.write(f'{i+1}. {r}\n')

    correct = 0
    wrong = 0
    for i, row in df.iterrows():
        print(f'Processing row {i+1}/{len(df)}...')

        # check out of bound the mark as wrong
        if i >= len(res):
            print('Out of bound detected!')
            print('Maybe SLM answer not generated properly...')
            wrong += 1
            continue

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

    res_dir = 'results'
    # res_file = 'llama_raw.txt' # For normal SLM without RAG
    # res_file = 'llama_logos.txt' # For normal Logos as RAG
    # res_file = 'llama_auxi.txt' # For AuxiDB + RAG
    # res_file = 'llama_multi_rag.txt' # For Multi RAG
    # res_file = 'llama_auxi_logos.txt'  # For AuxiLogos
    res_file = 'llama_auxi_db.txt' #For auxi_db only
    measure_slm_results(df, res_dir=res_dir, res_file=res_file)
