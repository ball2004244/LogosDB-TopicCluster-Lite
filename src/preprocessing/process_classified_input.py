import polars as pl
import os
import time

'''
This file preprocesses data before import into TopicCluster
It will takes user's csv, then generates 2 files: inputs.csv and topics.txt
'''

input_dir = "inputs/"
input_file = "inputs.csv"
out_csv = 'new_inputs.csv'
out_txt = 'metadata.txt'
input_path = os.path.join(input_dir, input_file)
out_csv_path = os.path.join(input_dir, out_csv)
out_txt_path = os.path.join(input_dir, out_txt)
BATCH_SIZE = 1000

# Set this flag based on your input file
HAS_HEADER = False  # Change to False if your CSV has no headers

start = time.perf_counter()
print(f'Reformatting {input_file}')

if HAS_HEADER:
    # For CSV with headers
    query = (
        pl.scan_csv(input_path)
        .select([
            'paragraph',
            'topic'
        ])
    )
else:
    # For CSV without headers
    query = (
        pl.scan_csv(input_path, has_header=False, new_columns=['paragraph', 'topic'])
        .select([
            'paragraph',
            'topic'
        ])
    )

query = (
    query
    .rename({
        'paragraph': 'content',
    })
    .filter(pl.col("topic").is_not_null())  # Filter out rows with empty topic value
)

# Replace space with dash in topic
query = query.with_columns(pl.col('topic').map_elements(lambda x: x.replace(' ', '-'), return_dtype=pl.Utf8).alias('topic'))

# Deduplicate paragraphs based on content
print('Deduplicating paragraphs...')
query = query.unique(subset=['content'], keep='first')

# Write new data to output csv
print(f'Save new data to {out_csv}...')
query.sink_csv(out_csv_path, batch_size=BATCH_SIZE, include_header=False)

# Write topics to txt file
print(f'Writing topics to {out_txt}...')
unique_topics = query.select('topic').collect()
topics = unique_topics['topic'].unique().to_list()
with open(out_txt_path, 'w') as f:
    for topic in topics:
        f.write(f'{topic}\n')

print(f'Takes {time.perf_counter() - start} seconds to process')