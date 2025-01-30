import os
import time
import glob
import json
import random
import base64
import logging
import argparse
import warnings
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
WORKING_DIRECTORY_BASE_NAME = f"tmp_fsc147_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"

parser = argparse.ArgumentParser(description='LVLM-COUNT: Enhancing the Counting Ability of Large Vision-Language Models. FSC-147 evaluation script.')
parser.add_argument('--dataset_path', type=str, default="data/fsc-147/test.json", required=False)
parser.add_argument('--image_base_path', type=str, default="data/fsc-147/images_384_VarV2", required=False)
parser.add_argument('--model_name', type=str, default="gpt-4o-2024-08-06", required=False)
parser.add_argument('--model_max_tokens', type=int, default=256, required=False)
parser.add_argument('--model_temperature', type=float, default=1.0, required=False)
parser.add_argument('--model_top_p', type=float, default=1.0, required=False)
parser.add_argument('--model_frequency_penalty', type=float, default=0.0, required=False)
parser.add_argument('--model_presence_penalty', type=float, default=0.0, required=False)
parser.add_argument('--mask_detection_box_threshold', type=float, default=0.1, required=False)
parser.add_argument('--mask_detection_text_threshold', type=float, default=0.1, required=False)
parser.add_argument('--mask_detection_iou_threshold', type=float, default=0.8, required=False)
parser.add_argument('--area_detection_box_threshold', type=float, default=0.15, required=False)
parser.add_argument('--area_detection_text_threshold', type=float, default=0.15, required=False)
parser.add_argument('--area_detection_iou_threshold', type=float, default=0.8, required=False)
parser.add_argument('--number_of_vertical_divides', type=int, default=-1, required=False)
parser.add_argument('--number_of_horizontal_divides', type=int, default=0, required=False)
parser.add_argument('--equivalent_size_division', default=False, action=argparse.BooleanOptionalAction)
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
logger.info("Number of Vertical Divides: %s", args.number_of_vertical_divides)
logger.info("Number of Horizontal Divides: %s", args.number_of_horizontal_divides)
logger.info("Equivalent Size Division: %s", args.equivalent_size_division)

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


from PIL import Image
def object_counter(question, object_of_interest_plural, image_path):
    image_pil = Image.open(image_path)
    image_arr = np.array(image_pil)
    colors = np.unique(np.reshape(image_arr, (-1, 4)), axis=0)
    n_objects = 0
    for color in colors:
        if not (
            np.array_equal(color, np.array([0, 0, 0, 0]))
            or
            np.array_equal(color, np.array([0, 0, 0, 255]))
        ):
            n_objects += 1
    return n_objects


def phrase_singulaizer(phrase):
    PROMPT_TEMPLATE = 'Provide the singular form of the phrase: "{phrase}"'
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = json.loads(
        client.chat.completions.create(
            model=args.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEMPLATE.format(phrase=phrase)}
                    ],
                }
            ],
            temperature=args.model_temperature,
            max_tokens=args.model_max_tokens,
            top_p=args.model_top_p,
            frequency_penalty=args.model_frequency_penalty,
            presence_penalty=args.model_presence_penalty,
            response_format={
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
            },
        )
        .choices[0]
        .message.content
    )
    logger.info(f"singular object of interest {response}")

    formatted_output = response["phrase"].strip("\"'. ")
    logger.debug(f"singular object of interest output: {formatted_output}")
    return formatted_output


def method_wrapper(question, object_of_interest, image_path, working_dirname):
    singular_form_of_object_of_interest = phrase_singulaizer(object_of_interest)

    object_aware_image_divider = ObjectAwareImageDivider(
        mask_detection_box_threshold=args.mask_detection_box_threshold,
        mask_detection_text_threshold=args.mask_detection_text_threshold,
        mask_detection_iou_threshold=args.mask_detection_iou_threshold,
        area_detection_box_threshold=args.area_detection_box_threshold,
        area_detection_text_threshold=args.area_detection_text_threshold,
        area_detection_iou_threshold=args.area_detection_iou_threshold,
        use_sam_masks=False
    )

    object_aware_image_divider.detect_area(
        image_path=image_path,
        gdino_prompt=f"{object_of_interest}.",
        output_folder=working_dirname
    )
    logger.info(
        "split image into %s area(s)",
        len(glob.glob(f"{working_dirname}/**_crops/*_crop_*.png")),
    )

    for area_image_path in sorted(
            glob.glob(f"{working_dirname}/**_crops/*_crop_*.png")
    ):
        output_dir = os.path.splitext(area_image_path)[0]
        os.makedirs(output_dir)
        object_aware_image_divider.divide_by_prompt(
            image_path=area_image_path,
            prompt="{}.".format(singular_form_of_object_of_interest),
            output_folder=output_dir,
            vertical_divides=args.number_of_vertical_divides,
            horizontal_divides=args.number_of_horizontal_divides,
            equi_size_div=args.equivalent_size_division
        )

    total_count = 0
    for subimage_path in sorted(
            glob.glob(f"{working_dirname}/**/**_masked_subimages/*.png", recursive=True)
    ):
        # with open(subimage_path, "rb") as f:
        #     subimage = f.read()
        count = object_counter(question, object_of_interest, subimage_path)
        logger.debug("counted %s for subimage %s", count, subimage_path)
        total_count += count

    logger.info("total count: %s for image %s", total_count, image_path)
    return total_count


def main():
    df = pd.read_json(args.dataset_path, orient="records")
    out_df = list()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        logger.debug("question: %s", row["question"])
        logger.debug("image_url: %s", row["image"])
        logger.debug("gold answer: %s", row["answer"])
        image_path = f"{args.image_base_path}/{row['image']}"

        working_dir = f"{WORKING_DIRECTORY_BASE_NAME}/{row['image']}"
        os.mkdir(working_dir)
        logger.debug("output directory: %s", working_dir)
        count = method_wrapper(
            row["question"], row["object_of_interest"], image_path, working_dir
        )
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
        f"Exact Accuracy: {(out_df['llm_count'].to_numpy() == out_df['answer'].to_numpy()).sum() / len(out_df)}"
    )
    logger.info(
        f"Mean Absolute Error: {sum(abs(out_df['llm_count'].to_numpy() - out_df['answer'].to_numpy())) / len(out_df)}"
    )
    logger.info(
        f"Root Mean Absolute Error: {(np.sqrt(sum((out_df['llm_count'].to_numpy() - out_df['answer'].to_numpy())**2)/len(out_df))).item()}"
    )


if __name__ == "__main__":
    main()
