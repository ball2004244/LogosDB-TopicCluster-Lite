from datasets import load_dataset
from call_rag import call_rag
from ollama import raw_call
from constants import OLLAMA_MODEL, RAG_PROMPT, SUFFIX_PROMPT, SUBJECT
import pandas as pd
import os
import time

'''
This file is for benchmarking the LogosDB as RAG for LLama 3.1 8B on MMLU dataset.

MUST Run LLama 3.1 8B First with the following command:
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

MUST Run VectorDB as well with the following command:
bash scripts/setup_sumdb.sh
'''


def benchmark_slm_rag(df: pd.DataFrame) -> None:
    print('Starting benchmarking on SLM + Logos RAG...')
    start = time.perf_counter()
    res_dir = 'results'
    res_file = 'llama_logos.txt'
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

        suffix_rag_prompt = '''You might find the following documents helpful to answer the question. If they are irrelevant, just ignore them and use your own reasoning.\n'''

        print(f'Querying RAG with question: {question}')
        print(f'Choices: {choices}')
        rag_prompt = RAG_PROMPT % (SUBJECT)
        # get top 3 results from RAG
        rag_results = call_rag(query=question, k=3)

        # build the prompt for the SLM
        for j, doc in enumerate(rag_results):
            suffix_rag_prompt += f'===== START DOCUMENT {j+1} =====\n'
            suffix_rag_prompt += f'{doc}\n'
            suffix_rag_prompt += f'===== END OF DOCUMENT {j+1} =====\n\n'

        suffix_rag_prompt += 'End of documents.\n'

        end_suffix_prompt = SUFFIX_PROMPT % (question, choices)
        # Only add the suffix prompt if there are results from RAG
        if rag_results:
            final_prompt = f'{rag_prompt}\n{suffix_rag_prompt}{end_suffix_prompt}'

        res = raw_call(final_prompt, model=OLLAMA_MODEL)
        with open(res_path, 'a') as f:
            f.write(f'Question {i+1}:\n')
            f.write(f'{res}\n')
            f.write('--------------------------\n')
            # f.write(f'Question: {question}\n')
            # f.write(f'Choices: {choices}\n')
            # f.write(f'Answer: {res}\n')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_slm_rag(df)
