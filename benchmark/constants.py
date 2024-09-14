OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
PROMPT = '''
You are an PhD expert in %s.
Answer this multiple-choice question below with accuracy. Carefully analyze the question and select the correct option. If unsure, choose the most likely answer based on your understanding..
Provide only the letter (A, B, C, D, or E) that corresponds to the correct answer. Do not provide any explanation or reasoning.
Example answer: A
'''

RAG_PROMPT = '''
You are an expert in %s.
You are given a question and a list of choices. Choose the best answer from the choices given.
'''

SUFFIX_PROMPT = '''
Question:
%s

Options:
%s
'''


SUBJECT = 'astronomy' #! CHANGE TO THE SUBJECT YOU WANT TO MEASURE
ANSWER_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}
