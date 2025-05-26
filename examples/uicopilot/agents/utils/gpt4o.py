import cv2
import io
import base64
import requests
import os
# from .config import *

def gpt4o(prompt, image, text):
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-13")
    ENDPOINT= f"https://{endpoint_base}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    image_base64 = base64.b64encode(img_str).decode('utf-8')

    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": [
                {
                    'type': 'text',
                    'text': prompt
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    'type': 'image',
                    'image': image_base64
                },
                {
                    'type': 'text',
                    'text': text
                }
            ],
        },
    ]

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    payload = {
        # "model": "gpt-4",
        "model": "gpt-4-turbo",
        "messages": PROMPT_MESSAGES,
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    response = requests.post(ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        return content
    else:
        return response.json()


