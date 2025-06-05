from typing import Iterator, List, Tuple, Union
from datetime import datetime
import os
import sqlite3
import time
import multiprocessing as mp
import pandas as pd

'''
This file contains the LogosCluster class, which is responsible for building a distributed system of SQLite databases.
'''

class LogosCluster:
    def __init__(self, data_dir: str='cluster_data') -> None:
        self.nodes = []
        self.data_dir = data_dir
        self.table_name = 'test_table'  # assume that the table in each node is the same

        self.input_file = None

    def set_metadata(self, metadata_file: str) -> None:
        '''
        Set metadata file for the cluster
        '''
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f'{metadata_file} not found, terminating the program')

        with open(metadata_file, 'r') as f:
            self.nodes = f.read().splitlines()

    def set_input_file(self, input_file: str) -> None:
        '''
        Set data file for the cluster
        '''
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found, terminating the program')

        self.input_file = input_file

    def build_cluster(self) -> List[str]:
        '''
        This function responsible for building a distributed system of SQLite databases.
        '''
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            for node in self.nodes:
                with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f'CREATE TABLE IF NOT EXISTS {self.table_name} (ID INTEGER PRIMARY KEY AUTOINCREMENT, Content TEXT, UpdatedAt DATETIME)')
                    conn.commit()

            return self.nodes
        except Exception as e:
            print(f'Error at LogosCluster build_cluster: {e}')
            return False

    def insert(self, data: List[Tuple[str, str]], node: str) -> bool:
        '''
        Insert data into 1 node
        '''
        try:
            for datum in data:
                content, updated_at = datum
                with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f'INSERT INTO {self.table_name} (Content, UpdatedAt) VALUES (?, ?)', (content, updated_at))
                    conn.commit()
            return True

        except Exception as e:
            print(f'Error at LogosCluster insert: {e}')
            return False

    def insert_batch(self, data: List[Tuple[str, str]], node: str) -> bool:
        '''
        Insert data into 1 node in batch
        '''
        try:
            with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    f'INSERT INTO {self.table_name} (Content, UpdatedAt) VALUES (?, ?)', data)
                conn.commit()
            return True

        except Exception as e:
            print(f'Error at LogosCluster insert_batch: {e}')
            return False

    def process_group(self, topic: str, data: pd.DataFrame) -> None:
        '''
        Process each group of data in parallel
        '''
        try:
            data['UpdatedAt'] = datetime.now().isoformat()

            # Convert DataFrame to list of tuples
            data_tuples = list(
                data[['content', 'UpdatedAt']].itertuples(index=False, name=None))

            self.insert_batch(data_tuples, topic)
        except Exception as e:
            print(f'Error at LogosCluster process_group: {e}')

    def auto_insert(self) -> bool:
        '''
        Auto insert data by chunk into a correct node in the system,

        Note that each topic serves as node name for now
        '''
        try:
            # First read the input file
            if self.input_file is None:
                raise FileNotFoundError('LogosCluster: Input file is not set')

            INPUT_CHUNK_SIZE = 10000

            # the input file only have 2 cols: content and topic
            headers = ['content', 'topic']
            count = 0
            with mp.Pool(mp.cpu_count()) as pool:
                for chunk in pd.read_csv(self.input_file, chunksize=INPUT_CHUNK_SIZE, usecols=[0, 1], header=None, names=headers):
                    print(f'Processing chunk {count}, CHUNK SIZE: {len(chunk)}')
                    start = time.perf_counter()

                    # Process 1 group at a time (single-process approach)
                    # print(f'Processing chunk of data sequentially...')
                    # self.process_group(chunk['topic'].iloc[0], chunk)
                    
                    # Process each group in parallel (multiprocessing approach)
                    # print(f'Utilzing {mp.cpu_count()} cores to process data...')
                    pool.starmap(self.process_group, [
                                 (topic, data) for topic, data in chunk.groupby('topic')])
                    count += 1
                    print(f'Finished processing chunk {count} in {time.perf_counter() - start:.2f} seconds')

            return True

        except Exception as e:
            print(f'Error at LogosCluster auto_insert: {e}')
            return False

    def query(self, _id: int, node: str) -> Union[Tuple[int, str, datetime], None]:
        '''
        Query specific data using ID from a specific node
        Input: node name and ID
        Output: Row schema (ID: int, Content: str, UpdatedAt: datetime)
        '''
        try:
            with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT * FROM {self.table_name} WHERE ID = ?', (_id,))
                result = cursor.fetchone()
            return result

        except Exception as e:
            print(f'Error at LogosCluster query: {e}')
            return None
        
    def query_by_ids(self, row_ids: List[int], node: str) -> List[Tuple[int, str, datetime]]:
        try:
            with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT * FROM {self.table_name} WHERE ID IN ({",".join([str(_id) for _id in row_ids])})')
                result = cursor.fetchall()
            return result
        except Exception as e:
            print(f'Error at LogosCluster query_by_ids: {e}')
            return []

    def query_all(self, node: str) -> List[Tuple[int, str, datetime]]:
        '''
        [WARNING] This function is not recommended for large dataset as leading to memory issues

        Query all data from a specific node
        Input: node name
        Output: List of rows (ID: int, Content: str, UpdatedAt: datetime)
        '''
        try:
            with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT * FROM {self.table_name}')
                result = cursor.fetchall()
            return result

        except Exception as e:
            print(f'Error at LogosCluster query_all: {e}')
            return []

    def query_chunk(self, node: str, CHUNK_SIZE: int=1000) -> Iterator[List[Tuple[int, str, datetime]]]:
        '''
        Query all data from a specific node in chunks
        Input: node name
        Output: List of rows (ID: int, Content: str, UpdatedAt: datetime)

        Example usage: 
        for chunk in cluster.query_chunk(node):
            for row in chunk:
                print(row)   
        '''
        try:
            with sqlite3.connect(f'{self.data_dir}/{node}.db') as conn:
                cursor = conn.cursor()
                cursor.execute(f'SELECT * FROM {self.table_name}')
                while True:
                    rows = cursor.fetchmany(CHUNK_SIZE)
                    if not rows:
                        break
                    yield rows
    
        except Exception as e:
            print(f'Error at LogosCluster query_chunk: {e}')
            return None