from typing import List, Tuple, Union
from datetime import datetime
import os
import sqlite3
import time
import pandas as pd


class LogosCluster:
    def __init__(self) -> None:
        self.nodes = []
        self.data_dir = 'cluster_data'
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
                    cursor.execute(f'INSERT INTO {self.table_name} (Content, UpdatedAt) VALUES (?, ?)', (content, updated_at))
                    conn.commit()
            return True
    
        except Exception as e:
            print(f'Error at LogosCluster insert: {e}')
            return False

    def auto_insert(self) -> bool:
        '''
        Auto insert data by chunk into a correct node in the system,

        Note that each topic serves as node name for now
        '''
        try:
            # TODO: Implement this function

            # First read the input file
            if self.input_file is None:
                raise FileNotFoundError('LogosCluster: Input file is not set')

            INPUT_CHUNK_SIZE = 1000

            # the input file only have 2 cols: content and topic
            headers = ['content', 'topic']
            for chunk in pd.read_csv(self.input_file, chunksize=INPUT_CHUNK_SIZE, usecols=[0, 1], header=None, names=headers):
                for topic, data in chunk.groupby('topic'):
                    data['UpdatedAt'] = datetime.now().isoformat()

                    # Convert DataFrame to list of tuples
                    data_tuples = list(
                        data[['content', 'UpdatedAt']].itertuples(index=False, name=None))

                    self.insert(data_tuples, topic)

            return True

        except Exception as e:
            print(f'Error at LogosCluster auto_insert: {e}')
            return False

    def query(self, _id: int, node: str) -> Union[Tuple[int, str, datetime], None]:
        '''
        Query data from a specific node,
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


def main():
    in_dir = 'inputs'
    metadata_file = 'metadata.txt'
    input_file = 'inp-10k.csv'  # ! Test on smaller dataset
    start = time.perf_counter()
    # First build the cluster
    cluster = LogosCluster()

    # Then set metadata and input file
    cluster.set_metadata(os.path.join(in_dir, metadata_file))
    cluster.set_input_file(os.path.join(in_dir, input_file))

    # Build the cluster
    cluster.build_cluster()

    # Populate the cluster with data
    cluster.auto_insert()

    print(f'Takes {time.perf_counter() - start} seconds to process')

if __name__ == '__main__':
    main()
