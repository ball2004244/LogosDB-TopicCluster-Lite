OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

PREFIX_PROMPT = '''You are an PhD expert in %s. Answer this multiple-choice question below with accuracy. Carefully analyze the question and select the correct option. If unsure, choose the most likely answer based on your understanding.\n'''

SUFFIX_PROMPT = '''You MUST answer using this pattern:\nQuestion Analyze:{Your analyse}\nDo you use the information given (Yes/No): {Your choice, explain why}\nReasoning: {Your reasoning and explanation of your answer}\nFinal Choice:\n\nExample answer:\nQuestion Analyze: This question is asking about the name of the 7th planet from the Sun.\nDo you use the information given (Yes/No): Yes because the information given are relevant to the question\nReasoning: Base on my understanding, it should be Uranus.\nFinal Choice: D. Uranus\n\nRemember that in your final choice, you should only include your choice from the choices given, no explanation (because that is what you have done in the "Reasoning" part). Also, you can only choose one choice from the choices given.\nQuestion:\n%s\nOptions:\n%s'''


SUBJECT = 'abstract_algebra'  # ! CHANGE TO THE SUBJECT YOU WANT TO MEASURE
ANSWER_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}
