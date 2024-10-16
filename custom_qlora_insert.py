from sumdb import SumDB
import pandas as pd
import time

'''
This code insert the presummarized data to Marqo

This code should be run once, and only after the data has been summarized
'''

print(f'INSERT Summarized Data to Marqo')
start = time.perf_counter()

# read csv file
print(f'[Step 1] Reading the summarized data from CSV...')
sum_data = pd.read_csv('debug/summarized_abstract_qlora.csv')
print(sum_data.head())

print(f'sum_data length:', len(sum_data))

# Add columns
sum_data.columns = ['row_id', 'summary', 'topic']


print(f'[Step 2] Inserting the summarized data to Marqo...')
CHUNK_SIZE = 128

sumdb = SumDB(port=8890)  # ! For AuxiLogosb Qlora Abstract
json_sum_data = sum_data.to_dict(orient='records')
# print(type(json_sum_data))
# print(type(json_sum_data[0]))

status = sumdb.insert(json_sum_data, CHUNK_SIZE)

if not status:
    print(f'Error inserting data to Marqo')
    exit(1)

print(f'Inserted {len(json_sum_data)} rows to Marqo')


print(
    f'FINISHED inserting in {time.perf_counter() - start:.2f} seconds (~{(time.perf_counter() - start) / 60:.2f} minutes)')
