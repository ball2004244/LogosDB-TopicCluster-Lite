from .base import benchmark_slm_rag
from constants import SUBJECT
from datasets import load_dataset
import pandas as pd

'''
This file is for benchmarking pure LLama 3.1 8B logical thinking on MMLU dataset.

MUST Run LLama 3.1 8B First with the following command:
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
'''


def benchmark_raw(df: pd.DataFrame, res_dir: str = 'results', res_file: str = 'llama_raw.txt', subject: str = SUBJECT) -> None:
    print(f'Raw Benchmarking on {subject}...')
    benchmark_slm_rag(df, res_dir=res_dir,
                      res_file=res_file, call_rag_func=None, subject=subject)
    print(f'Raw Benchmarking done!')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Topic: {df["subject"][0]}')
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_raw(df)
