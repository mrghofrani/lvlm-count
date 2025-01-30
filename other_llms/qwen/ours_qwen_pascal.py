import os
import re
import glob
import torch
import logging
import warnings
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from text_to_num import alpha2digit
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
WORKING_DIRECTORY_BASE_NAME = f"ours_qwen_pascal_{datetime.now().strftime('%d.%m.%Y-%H:%M:%S')}"
os.mkdir(WORKING_DIRECTORY_BASE_NAME)


df = pd.read_csv("output/output.csv")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    temperature=1.0,
    do_sample=True,
    top_p=0.95,
    top_k=64,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


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

    logger.info("model_name: %s", MODEL_NAME)

    out_df = list()
    messages = list()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']

        total_count = 0
        for image_path in sorted(glob.glob(f"output/{image_name}/**/**_subimages/*.png", recursive=True)):
            prompt = f"Count (if not possible, Estimate) the number of visible {row['category_name']} in the image. If you don't see any say zero. Report your final answer with digits inside **BOLDS**."
            logger.info("LLM input text prompt %s", prompt)
            logger.info("LLM input image prompt %s", image_path)

            while True:
                try:
                    message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(message)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")

                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    logger.info(f"LLM output %s", str(output_text[0]))
                    answer = re.search("\*\*(.+)\*\*", output_text[0]).group(1)
                    count = alpha2digit(answer, "en")
                    count = int(count)
                    break
                except Exception:
                    logger.exception("An exception Occurred")
            total_count += count

        logger.info(f"total count for %s was %d", image_name, total_count)
        out_df.append((index, row['image_name'], row['category_name'], row['answer'], total_count))
        pd.DataFrame(out_df, columns=["id", "image_name", "category_name", "answer", "llm_count"]).to_csv(f"{WORKING_DIRECTORY_BASE_NAME}/output{i}.csv", index=False)