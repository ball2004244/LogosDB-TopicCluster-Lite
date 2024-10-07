from datasets import load_dataset
from call_rag import call_auxi_db
from benchmark_base import benchmark_slm_rag
from constants import SUBJECT
import pandas as pd


def benchmark_auxi_db(df: pd.DataFrame, res_dir: str = 'results', res_file: str = 'llama_auxi_db.txt', subject: str = SUBJECT) -> None:
    '''
    This file only query from AuxiDB (SumDB) without follow-up call to LogosCluster.
    '''
    print('AuxiDB Benchmarking...')
    benchmark_slm_rag(df, res_dir=res_dir, res_file=res_file,
                      call_rag_func=call_auxi_db, subject=subject)
    print(f'AuxiDB Benchmarking done!')


if __name__ == '__main__':
    ds = load_dataset("cais/mmlu", SUBJECT)
    print('Keys:', ds.keys())

    # convert the dataset to a pandas dataframe
    df = pd.DataFrame(ds['test'])
    print(f'Length of test set: {len(df)}')
    print(f'Top 5 rows:\n{df.head()}')

    benchmark_auxi_db(df)
