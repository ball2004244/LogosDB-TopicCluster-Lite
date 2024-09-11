from constants import OLLAMA_URL
import requests

def raw_call(prompt: str, model="llama3.1:8b") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    res = response.json()['response']

    return res