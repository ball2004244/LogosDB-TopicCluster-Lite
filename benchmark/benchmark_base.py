from typing import Union
from datasets import load_dataset
from call_rag import call_rag
from ollama import raw_call
from constants import OLLAMA_MODEL, PREFIX_PROMPT, SUFFIX_PROMPT, SUBJECT
import pandas as pd
import os
import time
import string

'''
This file is for benchmarking the LogosDB as RAG for LLama 3.1 8B on MMLU dataset.

MUST Run LLama 3.1 8B First with the following command:
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

MUST Run VectorDB as well with the following command:
bash scripts/setup_sumdb.sh
'''


def benchmark_slm_rag(df: pd.DataFrame, res_dir: str = 'results', res_file: str = 'llama_logos.txt', subject: str = SUBJECT, call_rag_func: Union[None, callable] = None, k: int = 3) -> None:
    print('Starting benchmarking on SLM + Logos RAG...')
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

        rag_prompt = f'''Read through the following {k} pieces of information, retreive and make sure you understand them, then base on that information with your understanding to answer the question. If you think the information pieces given are irrelevant, then you can use only your understanding.\n'''

        print(f'Querying with question: {question}')
        print(f'Choices: {choices}')

        # if call_rag_func is not None, then call RAG
        # otherwise, treat it as raw SLM call
        rag_results = []
        if call_rag_func is not None:
            rag_results = call_rag_func(query=question, k=k)

        # build the prompt for the SLM
        for j, doc in enumerate(rag_results):
            rag_prompt += f'===== START OF INFORMATION {j+1} =====\n'
            rag_prompt += f'{doc}\n'
            rag_prompt += f'===== END OF INFORMATION {j+1} =====\n\n'

        if not rag_results:
            rag_prompt += 'NO INFORMATION PROVIDED. USE YOUR UNDERSTANDING TO ANSWER THE QUESTION.\n'

        start_prompt = PREFIX_PROMPT % (subject)
        end_prompt = SUFFIX_PROMPT % (question, choices)
        final_prompt = f'{start_prompt}{rag_prompt}{end_prompt}'

        # store prompt for debugging
        debug_dir = 'debug'
        with open(f'{debug_dir}/prompt.txt', 'a') as f:
            f.write(f'Prompt for question {i+1}\n')
            f.write(f'{final_prompt}\n')
            f.write('---------------------------\n')

        res = raw_call(final_prompt, model=OLLAMA_MODEL)
        with open(res_path, 'a') as f:
            f.write(f'Question {i+1}: {question}\n')
            f.write(f'Choices: {choices}\n')
            f.write(f'Answer: {res}\n')
            f.write('---------------------------------------\n')
    print(f'Benchmarking done in {time.perf_counter() - start} seconds.')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_slm_rag(df, call_rag_func=call_rag)
