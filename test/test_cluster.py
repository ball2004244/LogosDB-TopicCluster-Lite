import sqlite3
import os
'''
Test cluster in 1 node
'''

node_name = 'Astronomy'
table = 'test_table'
db_path = os.path.join('..', 'cluster_data', f'{node_name}.db')

# try connecting to the node and get all columns name
query = f'PRAGMA table_info({table})'

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute(query)
    columns = cursor.fetchall()
    
    print(columns)
    
print('Done')    
