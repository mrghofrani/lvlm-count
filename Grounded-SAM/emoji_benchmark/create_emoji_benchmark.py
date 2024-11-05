import os
import csv
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
L2 = 1
emojis_dir = "emojis/image/Apple"
full_emoji_csv = "emojis/full_emoji.csv"
emoji_categories_json = "emojis/emoji_classes.json"
emoji_renaming_json = "emojis/renaming_emojis.json"
output_dir = "tmp_benchmark"
NUM_TYPE = 6
MIN_NUMEL = 30  # Minimum number of elements of each type
MAX_NUMEL = 50  # Maximum number of elements of each type
PADDING_SIZE = 100
CANVAS_WIDTH = int(1024 * 2)
CANVAS_HEIGHT = int(1024 * 2)
ICON_WIDTH = 72
ICON_HEIGHT = 72
MEGA_CANVAS_WIDTH = CANVAS_WIDTH
MEGA_CANVAS_HEIGHT = CANVAS_HEIGHT
FINAL_CANVAS_WIDTH = 1024
FINAL_CANVAS_HEIGHT = 1024
ICON_SCALING_RATIO = 1.0
CANVAS_SCALING_RATIO = 1.0
SELECTION_LOSS_RATIO = 0.1

# Load emoji data
emoji_data = {}
with open(full_emoji_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header line
    for row in reader:
        emoji_id = row[0]  # First column
        name = row[3]   # Fourth column 
        emoji_data[name] = emoji_id

# Load emoji categories
with open(emoji_categories_json, 'r', encoding='utf-8') as f:
    categories = json.load(f)

with open(emoji_renaming_json, 'r', encoding='utf-8') as f:
    emoji_renaming = json.load(f)

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)


# Generate benchmark images and data
canvas_id = 1
question_id = 1
benchmark_data = list()
for class_, emoji_names in tqdm(categories.items()):
    emoji_selection_probability = np.ones(len(emoji_names))/len(emoji_names)
    mega_canvas_list = list()
    mega_canvas_data_list = list()
    for _ in range(L2):
        # Choose random emoji types, ensuring we don't request more than available
        selected_ids = np.random.choice(a=len(emoji_names), size=min(NUM_TYPE, len(emoji_names)), replace=False, p=emoji_selection_probability)
        chosen_types = [emoji_names[i] for i in selected_ids]

        new_emoji_selection_probability = list()
        for i, prob in enumerate(emoji_selection_probability):
            if i in selected_ids:
                new_emoji_selection_probability.append(prob * SELECTION_LOSS_RATIO)
            else:
                new_emoji_selection_probability.append(prob)
        emoji_selection_probability = np.array(new_emoji_selection_probability)/sum(new_emoji_selection_probability)

        # Generate random numbers for each type
        num_elements = [
            random.randint(MIN_NUMEL, MAX_NUMEL)
            for emoji_type in chosen_types
        ]

        # Create a blank image
        canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255, 255))

        placed_positions = []
        for i, emoji_type in enumerate(chosen_types):
            for _ in range(num_elements[i]):
                emoji_id = emoji_data[emoji_type]
                icon_path = os.path.join(emojis_dir, f"{emoji_id}.png")
                icon = Image.open(icon_path).convert("RGBA")
                icon = icon.resize((int(icon.width * ICON_SCALING_RATIO), int(icon.height * ICON_SCALING_RATIO)))

                # Find a random position for the icon (accounting for size)
                placed = False
                while not placed:
                    x = random.randint(icon.size[0], canvas.width - icon.size[0]) 
                    y = random.randint(icon.size[1], canvas.height - icon.size[1])
                    placed = True
                    # Check for overlap with existing icons
                    for existing_x, existing_y in placed_positions:
                        if (abs(x - existing_x) < icon.size[0] and 
                                abs(y - existing_y) < icon.size[1]):
                            placed = False
                            break

                # Paste the icon onto the image
                canvas.paste(icon, (x, y), icon)
                placed_positions.append((x, y))

        # Crop the image to remove excess whitespace (optional)
        canvas = canvas.crop(canvas.getbbox())
        mega_canvas_list.append(canvas)

        # Create image data for the JSON file
        canvas_data = {
            "types": chosen_types,
            "num_elements": num_elements,
        }
        mega_canvas_data_list.append(canvas_data)
        
    mega_canvas = Image.new("RGBA", (MEGA_CANVAS_WIDTH, MEGA_CANVAS_HEIGHT), (255, 255, 255, 255))
    mega_canvas.paste(mega_canvas_list[0], (0, 0), mega_canvas_list[0])
    # mega_canvas.paste(mega_canvas_list[1], (mega_canvas.width - mega_canvas_list[1].width - PADDING_SIZE, PADDING_SIZE), mega_canvas_list[1])
    # mega_canvas.paste(mega_canvas_list[2], (PADDING_SIZE, mega_canvas.height - mega_canvas_list[2].height - PADDING_SIZE), mega_canvas_list[2])
    # mega_canvas.paste(mega_canvas_list[3], (mega_canvas.width - mega_canvas_list[3].width - PADDING_SIZE, mega_canvas.height - mega_canvas_list[3].height - PADDING_SIZE), mega_canvas_list[3])

    # Save the image
    canvas_filename = f"image_{canvas_id:05d}.png"
    image_path = os.path.join(output_dir, "images", canvas_filename)
    mega_canvas = mega_canvas.resize((FINAL_CANVAS_WIDTH, FINAL_CANVAS_HEIGHT))
    mega_canvas.save(image_path)

    overal_type_count = dict()
    for data in mega_canvas_data_list:
        type_count = {type_:num for type_, num in zip(data['types'], data['num_elements'])}
        for key, value in type_count.items():
            if key not in overal_type_count:
                overal_type_count[key] = 0
            overal_type_count[key] += value

    for emoji_name, count in overal_type_count.items():
        benchmark_data.append({
            "id": question_id,
            "class": class_,
            "image_file": canvas_filename,
            "object_of_interest": emoji_renaming[emoji_name],
            "answer": count
        })
        question_id += 1
    canvas_id += 1


# Save benchmark data to JSON file
benchmark_json_path = os.path.join(output_dir, "benchmark_data.json")
with open(benchmark_json_path, 'w') as f:
    json.dump(benchmark_data, f, indent=4, ensure_ascii=False)
