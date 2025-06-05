from typing import List, Dict
from datasets import load_dataset
from src.core import SumDB
import time

'''
This code used to populate AuxiDB (Auxi VectorDb) with MMLU Auxi training data.
This allows query high quality data on similarity search.
'''


def preprocess_data(ds: dict) -> List[Dict[str, str]]:
    '''
    Preprocess the data to be inserted into the Auxi VectorDB.
    '''
    data = []
    for i, row in enumerate(ds['train']):
        correct_idx = row['answer']
        correct_choice = row['choices'][correct_idx]

        datum = f'Question: {row["question"]}\nAnswer: {correct_choice}\n'

        formatted_datum = {
            'row_id': i,
            'summary': datum,
            'topic': SUBJECT
        }
        data.append(formatted_datum)

    return data


if __name__ == '__main__':
    start = time.perf_counter()
    print(f'Load dataset...')
    SUBJECT = 'auxiliary_train'
    ds = load_dataset("cais/mmlu", SUBJECT, split='train')
    print(f'Loaded {len(ds)} rows from dataset')

    print(f'Preprocessing data...')
    data = preprocess_data(ds)
    print(f'Preprocessed {len(data)} rows')

    print(f'Inserting data into the VectorDB...')
    sumdb = SumDB('localhost', 8884)
    sumdb.insert(data)

    print(f'Inserted {len(data)} rows into the VectorDB for topic {SUBJECT}')
    print(
        f'Total ime taken: {time.perf_counter() - start:.2f} seconds (~ {time.perf_counter() - start:.2f} minutes)')
    print(f'Rate: {len(data)/(time.perf_counter() - start):.2f} rows/second')
