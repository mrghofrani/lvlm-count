import os
import time
import glob
import json
import base64
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import BadRequestError, NotFoundError, OpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_MODEL_NAME = "gpt-4o-2024-08-06"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MAX_TOKENS = 256
OPENAI_TEMPERATURE = 1.0
OPENAI_TOP_P = 1
OPENAI_FREQUENCY_PENALTY = 0
OPENAI_PRESENCE_PENALTY = 0
WORKING_DIRECTORY_BASE_NAME = f"tallyqa_complexbenchmark_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"

LLM = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

os.mkdir(WORKING_DIRECTORY_BASE_NAME)

def generate_content(logger, prompt, image=None, schema=None):
    if schema:
        if image:
            while True:
                try:
                    response = json.loads(
                        LLM.chat.completions.create(
                            model=OPENAI_MODEL_NAME,
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
                            temperature=OPENAI_TEMPERATURE,
                            max_tokens=OPENAI_MAX_TOKENS,
                            top_p=OPENAI_TOP_P,
                            frequency_penalty=OPENAI_FREQUENCY_PENALTY,
                            presence_penalty=OPENAI_PRESENCE_PENALTY,
                            response_format=schema,
                        )
                        .choices[0]
                        .message.content
                    )
                    return response
                except BadRequestError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
                except NotFoundError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
        else:
            while True:
                try:
                    response = json.loads(
                        LLM.chat.completions.create(
                            model=OPENAI_MODEL_NAME,
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
                            temperature=OPENAI_TEMPERATURE,
                            max_tokens=OPENAI_MAX_TOKENS,
                            top_p=OPENAI_TOP_P,
                            frequency_penalty=OPENAI_FREQUENCY_PENALTY,
                            presence_penalty=OPENAI_PRESENCE_PENALTY,
                            response_format=schema,
                        )
                        .choices[0]
                        .message.content
                    )
                    return response
                except BadRequestError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
                except NotFoundError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
    else:
        if image:
            while True:
                try:
                    response = (
                        LLM.chat.completions.create(
                            model=OPENAI_MODEL_NAME,
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
                            max_tokens=OPENAI_MAX_TOKENS,
                            temperature=OPENAI_TEMPERATURE,
                            top_p=OPENAI_TOP_P,
                            frequency_penalty=OPENAI_FREQUENCY_PENALTY,
                            presence_penalty=OPENAI_PRESENCE_PENALTY,
                        )
                        .choices[0]
                        .message.content
                    )
                    return response
                except BadRequestError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
                except NotFoundError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
        else:
            while True:
                try:
                    response = (
                        LLM.chat.completions.create(
                            model=OPENAI_MODEL_NAME,
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
                            max_tokens=OPENAI_MAX_TOKENS,
                            temperature=OPENAI_TEMPERATURE,
                            top_p=OPENAI_TOP_P,
                            frequency_penalty=OPENAI_FREQUENCY_PENALTY,
                            presence_penalty=OPENAI_PRESENCE_PENALTY,
                        )
                        .choices[0]
                        .message.content
                    )
                    return response
                except BadRequestError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)
                except NotFoundError:
                    logger.warning("a bad request execption occurred.", exc_info=True)
                    time.sleep(0.1)

def object_counter(logger, question, image):
    response = generate_content(
        logger=logger,
        prompt=question,
        image=image,
        schema={
            "type": "json_schema",
            "json_schema": {
                "name": "object_counting_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "additionalProperties": False,
                    "required": ["count"],
                },
            },
        }
    )
    logger.debug("llm count: %s", response["count"])
    return response["count"]


df = pd.read_csv("/u1/m2fetrat/GhCodes/visual-reasoning/result_final/tallyqa_complex_benchmark_ours_without_ChatGPT_areaDetection0.15/output.csv")
for i in range(2):
    logger = logging.getLogger(f'logger{i}')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(msecs)d %(levelname)s fn:%(funcName)s -- %(message)s")
    # Adding file handler 
    fh = logging.FileHandler(f"{WORKING_DIRECTORY_BASE_NAME}/log{i}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Adding stream handler 
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name != f"logger{i}":
            log_obj.disabled = True  # type: ignore
    warnings.filterwarnings("ignore")

    out_df = list()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        folder_name = f"{row['question_id']}.{row['question']}"
        rephrased_question = f"{row['question']} If you don't see any give zero."
        total_count = 0
        for subimage_path in sorted(glob.glob(f"/u1/m2fetrat/GhCodes/visual-reasoning/result_final/tallyqa_complex_benchmark_ours_without_ChatGPT_areaDetection0.15/{folder_name}/**/**_subimages/*.png", recursive=True)):
            with open(subimage_path, "rb") as f:
                subimage = f.read()
            count = object_counter(logger, rephrased_question, subimage)
            logger.debug("counted %s for subimage %s", count, subimage_path)
            total_count += count

        out_df.append((row['image'], row['answer'], row['data_source'], row['question'], row['image_id'], row['question_id'], row['issimple'], total_count))
        pd.DataFrame(out_df, columns=["image", "answer", "data_source", "question", "image_id", "question_id", "issimple", "llm_count"]).to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/output{i}.csv", index=False)