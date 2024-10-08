from datasets import load_dataset
from call_rag import call_auxi_train
from benchmark_rag import benchmark_slm_rag
from constants import SUBJECT
import pandas as pd


def benchmark_auxi_train(df: pd.DataFrame) -> None:
    '''
    This file only query from AuxiDB (SumDB) without follow-up call to LogosCluster.
    '''
    print('AuxiDB Benchmarking...')
    res_dir = 'results'
    res_file = 'llama_auxi_train.txt'
    benchmark_slm_rag(df, res_dir=res_dir, res_file=res_file,
                      call_rag_func=call_auxi_train)
    print(f'AuxiDB Benchmarking done!')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_auxi_train(df)
