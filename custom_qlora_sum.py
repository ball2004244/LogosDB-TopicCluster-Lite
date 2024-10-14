'''
This file is a temporary file to run summarize under no internet setting. Instead of run Summarize & Insert to Marqo at the same time,
we can run Summarize first, then Insert to Marqo later (because Marqo requires internet connection)

Flow: extract data from cluster -> summarize -> store in CSV file (So no use of SumDB for now)
'''
from cluster import LogosCluster
from qlora_abstract_sum import mass_qlora_abstract_sum
import time
import os
import datetime
import pandas as pd


def log(text: str, log_file: str = 'log.txt') -> None:
    current_time = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    text = f'{current_time} {text}\n'
    print(text)
    with open(log_file, 'a') as f:
        f.write(text)


log_path = os.path.join('debug', 'custom_qlora_sum.log')
# config cluster
in_dir = 'inputs'
metadata_file = 'metadata.txt'
input_file = 'input.csv'
logos_dir = 'auxi_logos'

log(f'Loading Config:', log_path)
log(f'Input dataset: {in_dir}/{input_file}', log_path)
log(f'Metadata file: {in_dir}/{metadata_file}', log_path)
log(f'Logos cluster save dir: {logos_dir}', log_path)

log(f'START RUNNING custom_qlora_sum.py...', log_path)
start = time.perf_counter()

# build cluster
log('[Step 1] Building the cluster...', log_path)
step_start = time.perf_counter()
cluster = LogosCluster(data_dir=logos_dir)
cluster.set_metadata(os.path.join(in_dir, metadata_file))
cluster.set_input_file(os.path.join(in_dir, input_file))
cluster.build_cluster()
log(f'Build cluster: {time.perf_counter() - step_start:.2f} seconds', log_path)

# NODE = 'Economics'  # summarize 1 node at a time
NODES = ['Philosophy', 'Chemistry', 'Psychology'] # summarize multi nodes
CHUNK_SIZE = 128  # this is the max number of rows to query from cluster
BATCH_SIZE = 16  # this is the max number of rows to summarize at a time

def summarize_nodes() -> None:
    for NODE in NODES:
        log('[Step 2] Summarizing 1 cluster node...', log_path)
        log(f'Config: NODE {NODE}, CHUNK {CHUNK_SIZE}, BATCH {BATCH_SIZE}', log_path)

        # store summarized data with 3 cols: row_id, summary, topic
        summarized_data = []

        count = 0
        for chunk in cluster.query_chunk(NODE, CHUNK_SIZE):
            step_start = time.perf_counter()
            log(f'Summarizing chunk {count}...', log_path)
            summaries = mass_qlora_abstract_sum([row[1] for row in chunk], BATCH_SIZE)

            # append summarized data to list
            for summary, row in zip(summaries, chunk):
                summarized_data.append({'row_id': row[0], 'summary': summary, 'topic': NODE})

            log(f'Finished summarizing chunk {count} in {time.perf_counter() - step_start:.2f} seconds', log_path)
            count += 1

        log(f'FINISHED SUMMARIZE NODE {NODE}', log_path)

        log(f'[Step 3] Saving summarized data to CSV...', log_path)
        df = pd.DataFrame(summarized_data)
        csv_path = os.path.join('debug', 'summarized_abstract_qlora.csv')

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=False, index=False)

        log(f'Summarized data saved to {csv_path}', log_path)
        log(f'Time elapsed: {time.perf_counter() - start:.2f} seconds (~{(time.perf_counter() - start)/60:.2f} minutes)', log_path)

summarize_nodes()
