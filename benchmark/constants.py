OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
PROMPT = '''
You are an PhD expert in %s.
Answer this multiple-choice question below with accuracy. Carefully analyze the question and select the correct option. If unsure, choose the most likely answer based on your understanding.
Provide only the letter corresponding to the correct answer, and only one option could be chosen. Do not provide any explanation or notes.

Example answer: A
'''

RAG_PROMPT = '''
You are an PhD expert in %s.
Answer this multiple-choice question below with accuracy. Carefully analyze the question and select the correct option. If unsure, choose the most likely answer based on your understanding.

'''

SUFFIX_PROMPT = '''
Question:
%s

Options:
%s
'''


SUBJECT = 'college_mathematics' #! CHANGE TO THE SUBJECT YOU WANT TO MEASURE
ANSWER_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}

