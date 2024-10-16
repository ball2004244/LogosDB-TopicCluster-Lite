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
    print(res)
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
    with open ('test/data_dig/wrong_answers.txt', 'w') as f:
        f.write(f'No\tRaw\tCorrect\n')
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
            with open('test/data_dig/wrong_answers_auxi.txt', 'a') as f:
                f.write(f'{i+1}.\t{raw_answer}  \t{answer}\n')
        elif ANSWER_MAP[raw_answer] == answer:
            correct += 1
        else:
            wrong += 1
            with open('test/data_dig/wrong_answers_auxi.txt', 'a') as f:
                f.write(f'{i+1}.\t{raw_answer}  \t{answer}\n')

    # save the result to a file
    stats_file = f'{res_file.split(".")[0]}_stats.txt'
    stats_path = os.path.join('test/data_dig', stats_file)
    with open(stats_path, 'w') as f:
        f.write(f'{topic_row}\n')
        f.write(
            f'Correct: {correct}, Wrong: {wrong}, Accuracy: {correct/len(df):.3f}')
    print(f'Save result to {stats_path}')


if __name__ == '__main__':
    SUBJECT = 'high_school_macroeconomics'  # ! CHANGE TO THE SUBJECT YOU WANT TO MEASURE
    ANSWER_MAP = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }
    ds = load_dataset("cais/mmlu", SUBJECT)
    df = pd.DataFrame(ds['test'])

    res_dir = 'results/auxi_db/high_school_macroeconomics/'
    res_file = 'llama_0.txt'
    measure_slm_results(df, res_dir=res_dir, res_file=res_file)
