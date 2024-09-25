import sqlite3
import os
'''
Test cluster in 1 node
'''
nodes_name = [
    'Biology', 
    'Mathematics', 
    'Psychology', 
    'History', 
    'Politics', 
    'Computer-Science', 
    'Geology', 
    'Chemistry', 
    'Astronomy', 
    'Physics', 
    'Literature', 
    'Economics', 
    'Art', 
    'Education', 
    'Philosophy'
]
def test_node(node_name):
    table = 'test_table'
    # cluster_dir = 'cluster_data'
    cluster_dir = 'auxi_logos'
    db_path = os.path.join('..', cluster_dir, f'{node_name}.db')

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
    return rows[0][0]

count = 0
for node_name in nodes_name:
    count += test_node(node_name)

print(f'Total nodes: {len(nodes_name)}')
print(f'Total rows in all nodes: {count}')
