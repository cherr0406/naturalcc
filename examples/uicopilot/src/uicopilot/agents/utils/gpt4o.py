import base64
import io
import os

import requests


def gpt4o(prompt, image, text, api_key: str | None = None, endpoint: str | None = None):
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
        "api-key": api_key if api_key else os.environ["API_KEY_OPENAI"],
    }
    payload = {
        # "model": "gpt-4",
        "model": "gpt-4-turbo",
        "messages": PROMPT_MESSAGES,
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 4096,
    }

    response = requests.post(endpoint if endpoint else os.environ["ENDPOINT_OPENAI"], headers=headers, json=payload)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        return content
    else:
        return response.json()


