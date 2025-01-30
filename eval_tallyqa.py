import os
import glob
import json
import time
import base64
import random
import logging
import warnings
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from src.object_aware_image_divider import ObjectAwareImageDivider
from openai import OpenAI
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openai.com/v1"
WORKING_DIRECTORY_BASE_NAME = f"tmp_tallyqabenchmark_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"

parser = argparse.ArgumentParser(description='LVLM-COUNT: Enhancing the Counting Ability of Large Vision-Language Models. TallyQA benchmark evaluation script.')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--image_base_path', type=str, required=True)
parser.add_argument('--model_name', type=str, default="gpt-4o-2024-08-06", required=False)
parser.add_argument('--model_max_tokens', type=int, default=256, required=False)
parser.add_argument('--model_temperature', type=float, default=1.0, required=False)
parser.add_argument('--model_top_p', type=float, default=1.0, required=False)
parser.add_argument('--model_frequency_penalty', type=float, default=0.0, required=False)
parser.add_argument('--model_presence_penalty', type=float, default=0.0, required=False)
parser.add_argument('--mask_detection_box_threshold', type=float, default=0.1, required=False)
parser.add_argument('--mask_detection_text_threshold', type=float, default=0.1, required=False)
parser.add_argument('--mask_detection_iou_threshold', type=float, default=0.8, required=False)
parser.add_argument('--area_detection_box_threshold', type=float, default=0.2, required=False)
parser.add_argument('--area_detection_text_threshold', type=float, default=0.2, required=False)
parser.add_argument('--area_detection_iou_threshold', type=float, default=0.8, required=False)

args = parser.parse_args()

LLM = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

os.mkdir(WORKING_DIRECTORY_BASE_NAME)

logging.basicConfig(
    format="%(asctime)s %(name)s %(msecs)d %(levelname)s fn:%(funcName)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(f"{WORKING_DIRECTORY_BASE_NAME}/log.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("logger")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != "logger":
        log_obj.disabled = True  # type: ignore
warnings.filterwarnings("ignore")


OBJECT_OF_INTEREST_DETERMINER_PROMPT = """
"{question}"
Summarize the above question in a phrase which starts with "Count of".
"""

logger.info("Run Hyperparameters")
logger.info("Dataset Path: %s", args.dataset_path)
logger.info("Image Base Path: %s", args.image_base_path)
logger.info("Model Name: %s", args.model_name)
logger.info("Model Max Tokens: %s", args.model_max_tokens)
logger.info("Model Temperature: %s", args.model_temperature)
logger.info("Model TOP_P: %s", args.model_top_p)
logger.info("Model Frequency Penalty: %s", args.model_frequency_penalty)
logger.info("Model Presence Penalty: %s", args.model_presence_penalty)
logger.info("Mask Detection Box Threshold: %s", args.mask_detection_box_threshold)
logger.info("Mask Detection Text Threshold: %s", args.mask_detection_text_threshold)
logger.info("Mask Detection IOU Threshold: %s", args.mask_detection_iou_threshold)
logger.info("Area Detection Box Threshold: %s", args.area_detection_box_threshold)
logger.info("Area Detection Text Threshold: %s", args.area_detection_text_threshold)
logger.info("Area Detection IOU Threshold: %s", args.area_detection_iou_threshold)


def generate_content(prompt, image=None, schema=None):
    if schema:
        if image:
            while True:
                try:
                    response = json.loads(
                        LLM.chat.completions.create(
                            model=args.model_name,
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
                            temperature=args.model_temperature,
                            max_tokens=args.model_max_tokens,
                            top_p=args.model_top_p,
                            frequency_penalty=args.model_frequency_penalty,
                            presence_penalty=args.model_presence_penalty,
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
                        LLM.chat.completions.create(
                            model=args.model_name,
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
                            temperature=args.model_temperature,
                            max_tokens=args.model_max_tokens,
                            top_p=args.model_top_p,
                            frequency_penalty=args.model_frequency_penalty,
                            presence_penalty=args.model_presence_penalty,
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
                        LLM.chat.completions.create(
                            model=args.model_name,
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
                            max_tokens=args.model_max_tokens,
                            temperature=args.model_temperature,
                            top_p=args.model_top_p,
                            frequency_penalty=args.model_frequency_penalty,
                            presence_penalty=args.model_presence_penalty,
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
                        LLM.chat.completions.create(
                            model=args.model_name,
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
                            max_tokens=args.model_max_tokens,
                            temperature=args.model_temperature,
                            top_p=args.model_top_p,
                            frequency_penalty=args.model_frequency_penalty,
                            presence_penalty=args.model_presence_penalty,
                        )
                        .choices[0]
                        .message.content
                    )
                    return response
                except Exception:
                    logger.warning("an execption occurred.", exc_info=True)
                    time.sleep(1)


def object_of_interest_determiner(question):
    llm_response = generate_content(
        OBJECT_OF_INTEREST_DETERMINER_PROMPT.format(question=question)
    )
    objects_of_interest = llm_response[
        llm_response.find("Count of") + len("Count of") :
    ].strip(". ")

    logger.debug("object_of_interest: %s", objects_of_interest)
    logger.debug("llm_response: %s", llm_response)
    return objects_of_interest


def object_counter(question, image):
    response = generate_content(
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


def phrase_singularizer(phrase):
    PROMPT_TEMPLATE = "Provide the singular form of the phrase: \"{phrase}\""
    response = generate_content(
        prompt = PROMPT_TEMPLATE.format(phrase=phrase),
        image = None,
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "singular_phrase_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"phrase": {"type": "string"}},
                    "additionalProperties": False,
                    "required": ["phrase"],
                },
            },
        }
    )
    logger.info(f"signular object of interest {response}")

    formatted_output = response["phrase"].strip("\"\'. ")
    logger.debug(f"signular object of interest output: {formatted_output}")
    return formatted_output


def method_wrapper(question, image_path, working_dirname):
    object_aware_image_divider = ObjectAwareImageDivider(
        mask_detection_box_threshold=args.mask_detection_box_threshold,
        mask_detection_text_threshold=args.mask_detection_text_threshold,
        mask_detection_iou_threshold=args.mask_detection_iou_threshold,
        area_detection_box_threshold=args.area_detection_box_threshold,
        area_detection_text_threshold=args.area_detection_text_threshold,
        area_detection_iou_threshold=args.area_detection_iou_threshold,
    )

    object_of_interest = object_of_interest_determiner(question)

    object_aware_image_divider.detect_area(
        image_path=image_path,
        gdino_prompt=object_of_interest,
        object_of_interest=object_of_interest,
        question=question,
        output_folder=working_dirname,
        chatgpt_enabled=False,
    )
    logger.info(
        "splitted image into %s area(s)",
        len(glob.glob(f"{working_dirname}/**_crops/*_crop_*.png")),
    )

    singular_form_of_object_of_interest = phrase_singularizer(object_of_interest)

    for area_image_path in sorted(
        glob.glob(f"{working_dirname}/**_crops/*_crop_*.png")
    ):
        output_dir = os.path.splitext(area_image_path)[0]
        os.makedirs(output_dir)
        object_aware_image_divider.divide_by_prompt(
            image_path=area_image_path,
            prompt=singular_form_of_object_of_interest,
            output_folder=output_dir,
            vertical_divides=-1,
            horizontal_divides=0,
        )

    rephrased_question = f"{question} If you don't see any give zero."

    total_count = 0
    for subimage_path in sorted(
        glob.glob(f"{working_dirname}/**/**_subimages/*.png", recursive=True)
    ):
        with open(subimage_path, "rb") as f:
            subimage = f.read()
        count = object_counter(rephrased_question, subimage)
        logger.debug("counted %s for subimage %s", count, subimage_path)
        total_count += count

    logger.info("total count: %s for image %s", total_count, area_image_path)
    return total_count


def main():
    df = pd.read_json(args.dataset_path, orient="records")
    out_df = list()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        logger.debug("question: %s", row["question_id"])
        logger.debug("question: %s", row["question"])
        logger.debug("image_url: %s", row["image"])
        logger.debug("gold answer: %s", row["answer"])
        image_path = f"{args.image_base_path}/{row['image']}"

        working_dir = f"{WORKING_DIRECTORY_BASE_NAME}/{row['question_id']}.{row['question']}"
        os.mkdir(working_dir)
        logger.debug("output directory: %s", working_dir)
        count = method_wrapper(row["question"], image_path, working_dir)
        out_df.append(
            row.tolist()
            + [
                count,
            ]
        )
        pd.DataFrame(out_df, columns=[df.columns.tolist() + ["llm_count",]]).to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/output.csv", index=False)

    out_df = pd.DataFrame(
        out_df,
        columns=[
            df.columns.tolist()
            + [
                "llm_count",
            ]
        ],
    )
    out_df.to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/output.csv", index=False)
    out_df.sort_index(inplace=True)

    logger.info(
        f"Exact Accuracy: {(out_df['llm_count'].to_numpy() == out_df['answer'].to_numpy()).sum()/len(out_df)}"
    )
    logger.info(
        f"Mean Absolute Error: {sum(abs(out_df['llm_count'].to_numpy() - out_df['answer'].to_numpy()))/len(out_df)}"
    )
    logger.info(
        f"Root Mean Absolute Error: {(np.sqrt(sum((out_df['llm_count'].to_numpy() - out_df['answer'].to_numpy())**2)/len(out_df))).item()}"
    )


if __name__ == "__main__":
    main()
