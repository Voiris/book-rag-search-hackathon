from ollama import chat

from config import OLLAMA_MODEL_NAME

def generate_answer(prompt: str):
    response = chat(model=OLLAMA_MODEL_NAME, messages=[{
        "role": "user",
        "content": prompt
    }])
    return response.message.content
