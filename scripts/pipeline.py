from src.core.cluster import LogosCluster
from src.core.sumdb import SumDB
import time
import os

'''
This file contains the main inferencing pipeline for the Logos project.
Flow: Dump documents into LogosCluster, summarize and store in SumDB, then query for similar content.
'''


def main():
    print('[Step 1] LAUNCHING LOGOS CLUSTER')
    in_dir = 'inputs'
    metadata_file = 'metadata.txt'
    input_file = 'input.csv'
    # input_file = 'inp-10k.csv'  # ! Test on smaller dataset
    # input_file = 'inp-100k.csv'

    # config logos cluster
    # logos_dir = 'cluster_data' #! Default cluster data directory
    logos_dir = 'auxi_logos'

    # config sumdb
    sum_db_host = 'localhost'
    # sum_db_port = 8882 #! Default port for SumDB
    # sum_db_port = 8885 #! For AuxiLogos Extract
    # sum_db_port = 8886 #! For AuxiLogos Abstract

    sum_db_port = 8890  # ! For AuxiLogosb Qlora Abstract

    print(f'Loading Config:')
    print(f'Input dataset: {in_dir}/{input_file}')
    print(f'Metadata file: {in_dir}/{metadata_file}')
    print(f'Logos cluster save dir: {logos_dir}')
    print(f'SumDB address: {sum_db_host}:{sum_db_port}')

    print('Start processing...')
    start = time.perf_counter()

    # First build the cluster
    cluster = LogosCluster(data_dir=logos_dir)

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
    sumdb = SumDB(host=sum_db_host, port=sum_db_port)
    print(f'Launch SumDB: {time.perf_counter() - step_start:.2f} seconds')

    # Step 3: Summarize all content from the cluster to SumDB with Summarizer
    CHUNK_SIZE = 128
    abstract_mode = True  # Toogle to switch between abstract and extractive summarization
    print('[Step 3] SUMMARIZING CLUSTER TO SUMDB')
    print(f'Enable abstract mode: {abstract_mode}')
    # step_start = time.perf_counter()
    # result = sumdb.summarize_cluster(cluster, CHUNK_SIZE=CHUNK_SIZE, abstract_mode=abstract_mode)

    # print(f'Summarize all cluster to SumDB: {time.perf_counter() - step_start:.2f} seconds')

    #! INSTEAD OF SUMMARIZING THE WHOLE CLUSTER, WE CAN SUMMARIZE A SINGLE NODE
    node_name = 'Economics'
    print(
        f'[Step 3] SUMMARIZING NODE {node_name} TO SUMDB, CHUNK SIZE: {CHUNK_SIZE}')
    step_start = time.perf_counter()
    if not abstract_mode:
        result = sumdb.summarize_node(
            node_name, cluster, CHUNK_SIZE=CHUNK_SIZE)
    else:
        result = sumdb.summarize_node_abstract(node_name, cluster,
                                               CHUNK_SIZE=CHUNK_SIZE)
    print(
        f'Summarize node to SumDB: {time.perf_counter() - step_start:.2f} seconds (~{(time.perf_counter() - start) / 60:.2f} minutes)')

    if not result:
        print('Unexpected error occurred during summarization, terminating...')
        return

    # Step 4: Smart Query, allow user search for similar content, first look up in SumDB, then in LogosCluster
    #! Check out smart_query.py & improved_query.py

    print(f'Total time: {time.perf_counter() - start:.2f} seconds to process')


if __name__ == '__main__':
    main()
