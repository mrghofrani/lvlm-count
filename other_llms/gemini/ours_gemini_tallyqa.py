
import os
import os
import glob
import json
import time
import logging
import warnings
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content


WORKING_DIRECTORY_BASE_NAME = f"ours_gemini_tallyqa_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"
os.mkdir(WORKING_DIRECTORY_BASE_NAME)


def upload_to_gemini(path, mime_type=None):
	"""Uploads the given file to Gemini.

	See https://ai.google.dev/gemini-api/docs/prompting_with_media
	"""
	file = genai.upload_file(path, mime_type=mime_type)
	print(f"Uploaded file '{file.display_name}' as: {file.uri}")
	return file


df = pd.read_csv("output/output.csv")
# Create the model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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

model_name = "gemini-1.5-pro"
model = genai.GenerativeModel(
	model_name=model_name,
	generation_config=generation_config,
)


for i in range(3):
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

    logger.info("model_name: %s", model_name)
    logger.info("generation_config %s", str(generation_config))

    out_df = list()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        folder_name = f"{row['question_id']}.{row['question']}"

        total_count = 0
        for image_path in sorted(glob.glob(f"output/{folder_name}/**/**_subimages/*.png", recursive=True)):
            prompt = row["question"]
            logger.info("LLM input text prompt %s", prompt)
            logger.info("LLM input image prompt %s", image_path)
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
                                prompt
                            ]
                        ).text
                    )
                    break
                except Exception:
                    logger.exception("An exception Occurred")
                    time.sleep(5)

            total_count += response['count']
            logger.info(f"LLM output %s", str(response))

        logger.info(f"total count for %s was %d", row['image'], total_count)
        out_df.append((row['image'], row['answer'], row['data_source'], row['question'], row['image_id'], row['question_id'], row['issimple'], total_count))
        pd.DataFrame(out_df, columns=["image", "answer", "data_source", "question", "image_id", "question_id", "issimple", "llm_count"]).to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/output{i}.csv", index=False)
