import pandas as pd

# Helpers
def word_count(doc: str) -> int:
    return len(doc.split())

def char_count(doc: str) -> int:
    return len(doc)

# Measure on summarized qlora dataset
df = pd.read_csv('debug/summarized_abstract_qlora.csv')
df.columns = ['row_id', 'summary', 'topic']

# run measure on summary col
df['word_count'] = df['summary'].apply(word_count)
df['char_count'] = df['summary'].apply(char_count)

total_words = df['word_count'].sum()
total_chars = df['char_count'].sum()
print(f'Total words: {total_words:,}')
print(f'Total chars: {total_chars:,}')

ORI_WORDS = 26394694
ORI_CHARS = 150181049

words_diff = ORI_WORDS - total_words
chars_diff = ORI_CHARS - total_chars

print(f'Words Improve: {words_diff:,} words ~ {words_diff / ORI_WORDS * 100}%')
print(f'Chars Improve: {chars_diff:,} words ~ {chars_diff / ORI_CHARS * 100}%')