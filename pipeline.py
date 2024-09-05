from cluster import LogosCluster
import time
import os

def main():
    print('[Step 1] LAUNCHING LOGOS CLUSTER!')
    in_dir = 'inputs'
    metadata_file = 'metadata.txt'
    # input_file = 'inp-10k.csv'  # ! Test on smaller dataset
    input_file = 'inp-100k.csv'
    
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

    # Populate the cluster with data
    step_start = time.perf_counter()
    print('Populating the cluster with data...')
    cluster.auto_insert()
    print(f'Populate cluster: {time.perf_counter() - step_start:.2f} seconds')

    # Step 2: Launch SumDB (VectorDB) with IndexTable

    # Step 3: Summarize all content from the cluster to SumDB with Summarizer

    # Step 4: Smart Query, allow user search for similar content, first look up in SumDB, then in LogosCluster    

    print(f'Total time: {time.perf_counter() - start:.2f} seconds to process')
    
if __name__ == '__main__':
    main()