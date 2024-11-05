import base64
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

load_dotenv()

OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MAX_TOKENS = 256
OPENAI_TEMPERATURE = 1.0
OPENAI_TOP_P = 1.0
WORKING_DIRECTORY_BASE_NAME = f"tmp_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"
OUTPUT_FILENAME = f"{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}.csv"
IMAGE_BASE_PATH = "data/genome"
DATASET_PATH = "data/tallyqa/test.json"

LLM = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

os.mkdir(WORKING_DIRECTORY_BASE_NAME)

logging.basicConfig(
    format="%(asctime)s %(name)s %(msecs)d %(levelname)s fn:%(funcName)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[logging.FileHandler(f"{WORKING_DIRECTORY_BASE_NAME}/log.log", encoding='utf-8'),
              logging.StreamHandler()],
)
logger = logging.getLogger("logger")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != "logger":
        log_obj.disabled = True  # type: ignore


QUESTION_ANSWERER_PROMPT = """
Answer the following question based on the given image.
"""


def generate_content(prompt, image=None):
    if image:
        while True:
            try:
                response = LLM.chat.completions.create(
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
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=OPENAI_MAX_TOKENS,
                    temperature=OPENAI_TEMPERATURE
                ).choices[0].message.content
                return response
            except BadRequestError:
                logger.warning("a bad request execption occurred.", exc_info=True)
    return (
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
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE
        )
        .choices[0]
        .message.content
    )


def object_counter(question, image):
    class OutputFormat(BaseModel):
        count: int

    completion = LLM.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": QUESTION_ANSWERER_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(image).decode('utf-8')}"
                        },
                    },
                ],
            },
        ],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
        top_p=OPENAI_TOP_P,
        response_format=OutputFormat,
    )
    output = completion.choices[0].message.parsed
    logger.debug("llm count: %s", output.count)
    return output.count


def method_wrapper(question, image_path, working_dirname):
    with open(image_path, "rb") as f:
        image = f.read()

    count = object_counter(question, image)
    logger.debug("counted %s for image %s", count, image_path)
    return count


def main():
    df = pd.read_json(DATASET_PATH, orient="records")
    out_df = list()
    _df = df[(~df["issimple"]) & (df["answer"] > 5)][:10]
    for _, row in tqdm(_df.iterrows(), total=len(_df)):
        logger.debug("question: %s", row['question'])
        logger.debug("image_url: %s", row['image'])
        logger.debug("gold answer: %s", row['answer'])
        image_path = f"{IMAGE_BASE_PATH}/{row['image']}"

        working_dir = f"{WORKING_DIRECTORY_BASE_NAME}/{row['question']}"
        os.mkdir(working_dir)
        logger.debug("output directory: %s", working_dir)
        count = method_wrapper(row["question"], image_path, working_dir)
        out_df.append(row.tolist() + [count,])

    out_df = pd.DataFrame(out_df, columns=[df.columns.tolist() + ["llm_count",]])
    out_df.to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/output.csv", index=False)
    out_df.sort_index(inplace=True)

    logger.info(f"Exact Accuracy: {(out_df['llm_count'].to_numpy() == out_df['answer'].to_numpy()).sum()/len(out_df)}")
    logger.info(f"Mean Absolute Error: {sum(abs(out_df['llm_count'].to_numpy() - out_df['answer'].to_numpy()))/len(out_df)}")  # noqa: E501


if __name__ == "__main__":
    main()
