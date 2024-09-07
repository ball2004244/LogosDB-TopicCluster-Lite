from cluster import LogosCluster
from sumdb import SumDB
import time
import os

'''
This file contains the main pipeline for the Logos project.
'''

def main():
    print('[Step 1] LAUNCHING LOGOS CLUSTER')
    in_dir = 'inputs'
    metadata_file = 'metadata.txt'
    input_file = 'input.csv'
    # input_file = 'inp-10k.csv'  # ! Test on smaller dataset
    # input_file = 'inp-100k.csv'
    
    print('Start processing...')
    start = time.perf_counter()
    
    # First build the cluster
    cluster = LogosCluster()

    # Set metadata
    step_start = time.perf_counter()
    cluster.set_metadata(os.path.join(in_dir, metadata_file))
    print(f'Set metadata: {time.perf_counter() - step_start:.2f} seconds')

    # Set input file
    step_start = time.perf_counter()
    cluster.set_input_file(os.path.join(in_dir, input_file))
    print(f'Set input file: {time.perf_counter() - step_start:.2f} seconds')

    # Build the cluster
    step_start = time.perf_counter()
    print('Building the cluster...')
    cluster.build_cluster()
    print(f'Build cluster: {time.perf_counter() - step_start:.2f} seconds')

    #! Only populate the cluster with data when needed
    # Populate the cluster with data
    # step_start = time.perf_counter()
    # print('Populating the cluster with data...')
    # cluster.auto_insert()
    # print(f'Populate cluster: {time.perf_counter() - step_start:.2f} seconds')

    # Step 2: Launch SumDB
    print('[Step 2] LAUNCHING SUMDB')
    step_start = time.perf_counter()
    sumdb = SumDB()
    print(f'Launch SumDB: {time.perf_counter() - step_start:.2f} seconds')

    # Step 3: Summarize all content from the cluster to SumDB with Summarizer
    # print('[Step 3] SUMMARIZING CLUSTER TO SUMDB')
    # step_start = time.perf_counter()
    # result = sumdb.summarize_cluster(cluster, CHUNK_SIZE=64)
    # print(f'Summarize cluster to SumDB: {time.perf_counter() - step_start:.2f} seconds')

    #! INSTEAD OF SUMMARIZING THE WHOLE CLUSTER, WE CAN SUMMARIZE A SINGLE NODE
    node_name = 'Astronomy'
    CHUNK_SIZE = 128
    print(f'[Step 3] SUMMARIZING NODE {node_name} TO SUMDB, CHUNK SIZE: {CHUNK_SIZE}')
    step_start = time.perf_counter()
    result = sumdb.summarize_node(node_name, cluster, CHUNK_SIZE=128)
    print(f'Summarize node to SumDB: {time.perf_counter() - step_start:.2f} seconds')

    if not result:
        print('Unexpected error occurred during summarization, terminating...')
        return

    # Step 4: Smart Query, allow user search for similar content, first look up in SumDB, then in LogosCluster    

    print(f'Total time: {time.perf_counter() - start:.2f} seconds to process')
    
if __name__ == '__main__':
    main()