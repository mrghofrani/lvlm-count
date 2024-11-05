
import os
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
from google.ai.generativelanguage_v1beta.types import content


WORKING_DIRECTORY_BASE_NAME = f"gemini_base_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"
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

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file


df = pd.read_json("/u1/m2fetrat/GhCodes/visual-reasoning/Grounded-SAM/emoji_benchmark/benchmark_one_canvas/benchmark_data.json")
# Create the model
genai.configure(api_key='AIzaSyCFCqMGCz5lPL8EnpBy4Zum7HjAnLrKsQk')
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_schema": content.Schema(
    type = content.Type.OBJECT,
    enum = [],
    required = ["count"],
    properties = {
      "count": content.Schema(
        type = content.Type.INTEGER,
      ),
    },
  ),
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
)

out_df = list()

for _, row in tqdm(df.iterrows(), total=len(df)):
    index = row['id']
    image_name = row['image_file']
    object_of_interest = row["object_of_interest"]
    folder_name = f"{index}.{object_of_interest}"

    image_path = f"/u1/m2fetrat/GhCodes/visual-reasoning/Grounded-SAM/emoji_benchmark/benchmark_one_canvas/images/{image_name}"

    while True:
        try:
            files = [
                upload_to_gemini(image_path, mime_type="image/png"),
            ]

            chat_session = model.start_chat(history=[])
            response = json.loads(
                chat_session.send_message(
                    [
                        files[0],
                        f"How many {row['object_of_interest']} are visibile in the image?"
                    ]
                ).text
            )
            break
        except Exception:
            logger.exception("An exception Occurred")
            time.sleep(5)

    total_count = response['count']
    logger.info(f"LLM exact count for {image_path} was {response}")
    out_df.append((row['id'], row['image_file'], row['object_of_interest'], row['answer'], total_count))
    pd.DataFrame(out_df, columns=["id", "image_file", "object_of_interest", "answer", "llm_count"]).to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/gemini-output.csv", index=False)
