from datasets import load_dataset
from call_rag import call_auxi_logos
from benchmark_rag import benchmark_slm_rag
from constants import SUBJECT
import pandas as pd

'''
This file is for benchmarking the LogosDB as RAG for LLama 3.1 8B on MMLU dataset.

MUST Run LLama 3.1 8B First with the following command:
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

MUST Run VectorDB as well with the following command:
bash scripts/setup_sumdb.sh
'''


def benchmark_auxi_logos(df: pd.DataFrame) -> None:
    '''
    This file for querying normal LogosDB as RAG but populated with Auxiliary Dataset.
    '''
    print('AuxiLogos Benchmarking...')
    res_dir = 'results'
    res_file = 'llama_auxi_logos.txt'
    benchmark_slm_rag(df, res_dir=res_dir, res_file=res_file,
                      call_rag_func=call_auxi_logos)
    print(f'AuxiLogos Benchmarking done!')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_auxi_logos(df)
