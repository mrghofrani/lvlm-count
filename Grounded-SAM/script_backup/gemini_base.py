import os
import glob
import json
import time
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import google.generativeai as genai


WORKING_DIRECTORY_BASE_NAME = f"old_api_gemini_base_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"
os.mkdir(WORKING_DIRECTORY_BASE_NAME)


logging.basicConfig(
    format="%(asctime)s %(name)s %(msecs)d %(levelname)s fn:%(funcName)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(f"{WORKING_DIRECTORY_BASE_NAME}/gemini-log.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("logger")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != "logger":
        log_obj.disabled = True  # type: ignore


genai.configure(api_key='AIzaSyCFCqMGCz5lPL8EnpBy4Zum7HjAnLrKsQk')
model = genai.GenerativeModel('models/gemini-1.5-pro')


df = pd.read_json("/u1/m2fetrat/GhCodes/visual-reasoning/Grounded-SAM/data/fsc-147/test.json")
out_df = list()
for _, row in tqdm(df.iterrows(), total=len(df)):
    index = row['id']
    image_path = f"/u1/m2fetrat/GhCodes/visual-reasoning/Grounded-SAM/data/fsc-147/images_384_VarV2/{row['image']}"
    object_of_interest = row["object_of_interest"]

    while True:
        try:
            response = json.loads(
                model.generate_content(
                    [
                        row['question'],
                        Image.open(image_path)
                    ],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer"
                                }
                            },
                            "required": [
                                "count"
                            ]
                        }
                    ),
                ).candidates[0].content.parts[0].text
            )
            break
        except Exception:
            logger.exception("An exception Occurred")
            time.sleep(5)

    total_count = response['count']
    logger.info(f"LLM exact count for {image_path} was {response}")

    out_df.append((row['id'], row['image'], row['object_of_interest'], row['question'], row['answer'], total_count))
    pd.DataFrame(out_df, columns=["id", "image", "object_of_interest", "question", "answer", "llm_count"]).to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/gemini-output.csv", index=False)
