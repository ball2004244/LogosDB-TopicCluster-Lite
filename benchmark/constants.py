OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
PROMPT = '''
You are an expert in %s.
You are given a question and a list of choices. Choose the best answer from the choices given.
The initial choices come as a list of string, but the desired answer must be only "A", "B", "C", "D", or "E". 
No yapping. No explain. No nothing. Just the letter.
'''

RAG_PROMPT = '''
You are an expert in %s.
You are given a question and a list of choices. Choose the best answer from the choices given.
'''

SUFFIX_PROMPT = '''
Here is your question:
%s

Here are the choices:
%s

Example answer:
A

Your answer:
'''


SUBJECT = 'astronomy' #! CHANGE TO THE SUBJECT YOU WANT TO MEASURE
ANSWER_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}