import json
import time
import base64
import logging
from openai import OpenAI


logger = logging.getLogger("logger")

class GPT4oClient:
    def __init__(
        self, 
        api_key,
        model_name="gpt-4o-2024-08-06",
        temperature=1.0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    ):
        self.model_name = model_name
        self.temperature= temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.openai_client = OpenAI(api_key=api_key)
    
    def generate_content(self, prompt, image=None, schema=None):
        if schema:
            if image:
                while True:
                    try:
                        response = json.loads(
                            self.openai_client.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{base64.b64encode(image).decode('utf-8')}"
                                                },
                                            },
                                            {
                                                "type": "text",
                                                "text": prompt,
                                            },
                                        ],
                                    }
                                ],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                top_p=self.top_p,
                                frequency_penalty=self.frequency_penalty,
                                presence_penalty=self.presence_penalty,
                                response_format=schema,
                            )
                            .choices[0]
                            .message.content
                        )
                        return response
                    except Exception:
                        logger.warning("an execption occurred.", exc_info=True)
                        time.sleep(1)
            else:
                while True:
                    try:
                        response = json.loads(
                            self.openai_client.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": prompt,
                                            },
                                        ],
                                    }
                                ],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                top_p=self.top_p,
                                frequency_penalty=self.frequency_penalty,
                                presence_penalty=self.presence_penalty,
                                response_format=schema,
                            )
                            .choices[0]
                            .message.content
                        )
                        return response
                    except Exception:
                        logger.warning("an execption occurred.", exc_info=True)
                        time.sleep(1)
        else:
            if image:
                while True:
                    try:
                        response = (
                            self.openai_client.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": prompt,
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{base64.b64encode(image).decode('utf-8')}"
                                                },
                                            },
                                        ],
                                    }
                                ],
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                frequency_penalty=self.frequency_penalty,
                                presence_penalty=self.presence_penalty,
                            )
                            .choices[0]
                            .message.content
                        )
                        return response
                    except Exception:
                        logger.warning("an execption occurred.", exc_info=True)
                        time.sleep(1)
            else:
                while True:
                    try:
                        response = (
                            self.openai_client.chat.completions.create(
                                model=self.model_name,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": prompt,
                                            }
                                        ],
                                    }
                                ],
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                frequency_penalty=self.frequency_penalty,
                                presence_penalty=self.presence_penalty,
                            )
                            .choices[0]
                            .message.content
                        )
                        return response
                    except Exception:
                        logger.warning("an execption occurred.", exc_info=True)
                        time.sleep(1)
