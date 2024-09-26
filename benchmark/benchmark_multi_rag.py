from datasets import load_dataset
from call_rag import call_multi_rag
from benchmark_rag import benchmark_slm_rag
from constants import SUBJECT
import pandas as pd


'''
This file measure the performance of MultiRAG Architecture

Flow:
Query -> AuxiDB -> SumDB -> LogosCluster -> ParaDB -> LLama -> Answer

Dataset info:
AuxiDB: MMLU Auxiliary dataset
SumDB: Summarized info from WikiSum dataset
LogosCluster: Full docs from WikiSum
ParaDB: Splitted docs from WikiSum
'''


def benchmark_multi_rag(df: pd.DataFrame) -> None:
    print('MULTI RAG Benchmarking...')
    res_dir = 'results'
    res_file = 'llama_multi_rag.txt'
    benchmark_slm_rag(df, res_dir=res_dir, res_file=res_file,
                      call_rag_func=call_multi_rag)
    print(f'MULTI RAG Benchmarking done!')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_multi_rag(df)
