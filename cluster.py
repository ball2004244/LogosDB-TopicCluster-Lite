from typing import List, Tuple, Union
from datetime import datetime
import os
import sqlite3

class LogosCluster:
    def __init__(self, metadata_file: str) -> None:
        # check if the metadata file exists
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f'{metadata_file} not found, terminating the program')

        self.nodes = []
        with open(metadata_file, 'r') as f:
            self.nodes = f.read().splitlines()
        self.data_dir = 'cluster_data'
        self.table_name = 'test_table' # assume that the table in each node is the same

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
                    cursor.execute(f'CREATE TABLE {self.table_name} (ID int, Content text, UpdatedAt datetime)')
                    conn.commit()

            return self.nodes
        except Exception as e:
            print(f'Error: {e}')
            return False

    def insert(self, data: List[Tuple[int, str, datetime]], node: str) -> bool:
        '''
        Insert data into 1 node
        '''
        try:
            for datum in data:
                _id, content, updated_at, topic = datum
                with sqlite3.connect(f'{self.data_dir}/{topic}.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'INSERT INTO {self.table_name} (ID, Content, UpdatedAt) VALUES (?, ?, ?)', (_id, content, updated_at))
                    conn.commit()
            return True

        except Exception as e:
            print(f'Error: {e}')
            return False

    def auto_insert(self, data: List[Tuple[int, str, datetime, str]]) -> bool:
        '''
        auto insert data into a correct node in the system,
        Input Row schema (ID: int, Content: str, UpdatedAt: datetime, topic: str)
        Inserted Row schema (ID: int, Content: str, UpdatedAt: datetime)

        Note that each topic serves as node name for now

        Also assume that the inserted table named 'test_table'
        '''
        try:
            # TODO: Implement this function
            return True

        except Exception as e:
            print(f'Error: {e}')
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
                cursor.execute(f'SELECT * FROM {self.table_name} WHERE ID = ?', (_id,))
                result = cursor.fetchone()
            return result

        except Exception as e:
            print(f'Error: {e}')
            return None

if __name__ == '__main__':
    metadata_file = 'metadata.txt'
    cluster = LogosCluster(metadata_file)
    cluster.build_cluster()
