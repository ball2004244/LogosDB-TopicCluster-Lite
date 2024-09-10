import sqlite3
import os
'''
Test cluster in 1 node
'''

node_name = 'Chemistry'
table = 'test_table'
db_path = os.path.join('..', 'cluster_data', f'{node_name}.db')

print(f'Getting columns name from {node_name}...')
query = f'PRAGMA table_info({table})'
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute(query)
    columns = cursor.fetchall()
    
    print(columns)

print(f'Counting rows in {node_name}...')
query = f'SELECT COUNT(*) FROM {table}'
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()

print(f'Number of rows in {node_name}: {rows[0][0]}')

print('Done')    
