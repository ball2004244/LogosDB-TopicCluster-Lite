import pandas as pd

in_file = 'inputs/input.csv'

df = pd.read_csv(in_file)

# assign header to df (content, topic)
df.columns = ['content', 'topic']

# get row freq of each topic
print(f'Number of rows by topic:')
print(df['topic'].value_counts())

# get num chunk, assume each chunk has 128 rows
num_chunk = len(df) // 128
print(f'Number of chunk: {num_chunk}')

# get num chunk by topic
num_chunk_by_topic = df.groupby('topic').size().sort_values(ascending=False) / 128
print(num_chunk_by_topic)
