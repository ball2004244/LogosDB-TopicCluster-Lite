import pandas as pd

in_file = '../inputs/input.csv'
out_file = '../inputs/new_input.csv'
df = pd.read_csv(in_file, header=None)

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

headers = ['content', 'topic']

# assign header to df
df.columns = headers
df['topic'] = df['topic'].apply(lambda x: x if x in nodes_name else x.replace(' ', '-'))

# save the new DataFrame to a new file without header
df.to_csv(out_file, index=False, header=False)
print('Done')