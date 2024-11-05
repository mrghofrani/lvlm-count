import base64
import json
import logging
import os
import random
from io import BytesIO

import cv2
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torchvision.transforms.functional as F  # type: ignore
from dotenv import load_dotenv
from openai import OpenAI
from gradio_client import Client
from PIL import Image, ImageDraw, ImageFont
from Real_ESRGAN.superResolution import SuperResolution
from sklearn.cluster import MeanShift  # type: ignore
from sklearn.cluster import KMeans

load_dotenv()


RANDOM_SEED = 42
WHITE_BW_COLOR = 255
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILLER_PINK_RGBA_COLOR = (255, 145, 175, 255)

np.random.seed(RANDOM_SEED)
logger = logging.getLogger("logger")


class ImageDivider:
    def __init__(
        self,
        image_path,
        mask_image,
        output_folder,
        vertical_divides,
        horizontal_divides,
        superRes,
        equi_size_div,
        add_background=False,
    ):
        self.image_path = image_path
        self.mask_image = mask_image
        self.output_folder = output_folder
        self.num_vertical_divides = vertical_divides
        self.num_horizontal_divides = horizontal_divides
        self.equi_size_div = equi_size_div
        self.paths = []
        self.sample_ratio = 0.015
        self.min_number_of_masks_to_divide_image = 6
        self.pad_output_height = 800
        self.pad_output_width = 1000
        self.black_white_mask = np.array(self.mask_to_black_white(self.mask_image))
        Image.fromarray(self.black_white_mask).save(f"{self.output_folder}/bw_mask.png")

        self.add_background = add_background
        # Extract the image name without the extension
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Ensure output folder and subfolder exist
        self.subimages_folder = os.path.join(
            self.output_folder, f"{self.image_name}_subimages"
        )
        os.makedirs(self.subimages_folder, exist_ok=True)
        self.subimages_before_super_resolution_folder = os.path.join(
            self.output_folder, f"{self.image_name}_subimages_before_super_resolution"
        )
        os.makedirs(self.subimages_before_super_resolution_folder, exist_ok=True)

        self.sr = SuperResolution(model_name="RealESRGAN_x4plus", face_enhance=True)
        self.superRes = superRes

    def mask_to_black_white(self, mask_image):
        mask_gray = cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2GRAY)
        _, mask_black_and_white = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
        return Image.fromarray(255 - mask_black_and_white)

    def divide_image(self):
        """
        Calculates the start and end points for cutting lines to divide an image into subimages.

        Returns:
            A tuple containing two lists:
                - vertical_lines: A list of tuples, each containing the start and end points of a vertical cutting line.
                - horizontal_lines: A list of tuples, each containing the start and end points of a horizontal cutting line.
        """  # noqa: E501
        image = np.array(self.black_white_mask)
        height, width = image.shape

        # Calculate the spacing between division points
        vertical_spacing = width // (self.num_vertical_divides + 1)
        horizontal_spacing = height // (self.num_horizontal_divides + 1)

        # Generate the cutting line coordinates (adjusting end points for 0-based indexing)
        vertical_lines = [
            ((x * vertical_spacing, 0), (x * vertical_spacing, height - 1))
            for x in range(1, self.num_vertical_divides + 1)
        ]
        horizontal_lines = [
            ((0, y * horizontal_spacing), (width - 1, y * horizontal_spacing))
            for y in range(1, self.num_horizontal_divides + 1)
        ]

        self._division_line_visualizer(
            x_dividing_lines=vertical_lines,
            y_dividing_lines=horizontal_lines,
        )
        return vertical_lines, horizontal_lines

    def cluster_projected_points(self, points):
        ms = MeanShift()
        ms.fit(points)
        return ms.labels_, ms.cluster_centers_

    def find_dividing_lines(self, points, labels, axis):
        sorted_clusters = []
        for label in np.unique(labels):
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points[:, axis])
            sorted_clusters.append((label, cluster_center))

        sorted_clusters.sort(key=lambda x: x[1])

        dividing_lines = []
        for i in range(len(sorted_clusters) - 1):
            current_label, current_center = sorted_clusters[i]
            next_label, next_center = sorted_clusters[i + 1]

            current_max = np.max(points[labels == current_label][:, axis])
            next_min = np.min(points[labels == next_label][:, axis])

            dividing_line = (current_max + next_min) // 2
            dividing_lines.append(dividing_line)

        return dividing_lines

    def _division_line_visualizer(
        self, x_dividing_lines, y_dividing_lines, samples=None
    ):
        plt.imshow(self.black_white_mask, cmap="Greys")
        if samples is not None:
            plt.scatter(samples[:, 1], samples[:, 0], c="red", s=10)

        for start, end in x_dividing_lines:
            plt.axline(start, end, color="blue", linestyle="--")
        for start, end in y_dividing_lines:
            plt.axline(start, end, color="green", linestyle="--")

        plt.title("Original Image with Dividing Lines")
        plt.savefig(f"{self.output_folder}/division_lines.png")
        plt.clf()

    def cluster_divide_image(self):
        """
        Calculates the start and end points for cutting lines to divide an image into subimages.

        Args:
            image_pil: A PIL image representin gthe masks of objects of interest in the image

        Returns:
            A tuple containing two lists:
                - vertical_lines: A list of tuples, each containing the start and end points of a vertical cutting line.
                - horizontal_lines: A list of tuples, each containing the start and end points of a horizontal cutting line.
        """  # noqa: E501

        mask_pixel_locations = np.argwhere(self.black_white_mask == 0)
        number_of_samples = int(self.sample_ratio * len(mask_pixel_locations))
        logger.debug("Number of samples for clustering: %s", number_of_samples)
        samples = mask_pixel_locations[
            np.random.randint(
                len(mask_pixel_locations),
                size=number_of_samples,
            )
        ]

        x_labels, _ = self.cluster_projected_points(samples[:, 1].reshape(-1, 1))
        y_labels, _ = self.cluster_projected_points(samples[:, 0].reshape(-1, 1))

        x_dividing_lines = self.find_dividing_lines(samples, x_labels, axis=1)
        y_dividing_lines = self.find_dividing_lines(samples, y_labels, axis=0)

        height, width = self.black_white_mask.shape
        vertical_lines = [((x, 0), (x, height - 1)) for x in x_dividing_lines]
        horizontal_lines = [((0, y), (width - 1, y)) for y in y_dividing_lines]

        self._division_line_visualizer(
            samples=samples,
            x_dividing_lines=vertical_lines,
            y_dividing_lines=horizontal_lines,
        )

        return vertical_lines, horizontal_lines

    def create_graph_from_image(self):
        """Creates a graph representation of the image for pathfinding.

        Returns:
            networkx.Graph: A graph where nodes are walkable pixels and edges
                            connect neighbors.
        """
        height, width = self.black_white_mask.shape[:2]
        graph = nx.Graph()

        for y in range(height):
            for x in range(width):
                # If pixel is white (walkable)
                if self.black_white_mask[y, x] == 255:
                    node_index = y * width + x
                    graph.add_node(node_index)

                    # Add edges to 8-connected neighbors
                    for dx, dy in [
                        (0, 1),
                        (1, 0),
                        (0, -1),
                        (-1, 0),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ]:
                        neighbor_x, neighbor_y = x + dx, y + dy
                        if (
                            0 <= neighbor_x < width
                            and 0 <= neighbor_y < height
                            and self.black_white_mask[neighbor_y, neighbor_x] == 255
                        ):
                            neighbor_index = neighbor_y * width + neighbor_x
                            graph.add_edge(node_index, neighbor_index)

        return graph

    def find_path_a_star(self, graph, start, end):
        """Finds the shortest path between two points using A* search.

        Args:
            graph (networkx.Graph): The graph representation of the image.
            start (tuple): (x, y) coordinates of the starting point.
            end (tuple): (x, y) coordinates of the end point.
            width (int): Width of the original image (for coordinate conversion).

        Returns:
            list: A list of (x, y) coordinates representing the path, or None
                if no path is found.
        """
        _, width = self.black_white_mask.shape

        start_node = start[1] * width + start[0]
        end_node = end[1] * width + end[0]

        def manhattan_distance(u, v):
            """Calculates Manhattan distance between two nodes."""
            ux, uy = u % width, u // width
            vx, vy = v % width, v // width
            return abs(ux - vx) + abs(uy - vy)

        try:
            path_nodes = nx.astar_path(
                graph, start_node, end_node, heuristic=manhattan_distance
            )
            path_coordinates = [(node % width, node // width) for node in path_nodes]
            return path_coordinates
        except nx.NetworkXException:
            return None  # No path found

    def connect_points_with_paths(self):
        if self.equi_size_div == True:
            vertical_lines, horizontal_lines = self.divide_image()
        else:
            vertical_lines, horizontal_lines = self.cluster_divide_image()

        graph = self.create_graph_from_image()
        if self.num_horizontal_divides != 0:
            division_lines = vertical_lines + horizontal_lines
        else:
            division_lines = vertical_lines

        self.paths = []
        for start, end in division_lines:
            logger.debug("Path start point: %s, Path end point: %s", start, end)
            path = self.find_path_a_star(graph, start, end)
            if path:
                self.paths.append(path)
            else:
                logger.warning(f"Warning: No path found between {start} and {end}")
        return self.paths

    def split_image_by_paths(self):
        """
        Splits an image into subimages based on provided paths.

        Returns:
            A list of subimages, each as a NumPy array.
        """
        image = np.array(Image.open(self.image_path))
        ##These lines temporarily added by MFQ just to plot paths on image-start
        # image_copy = np.array(Image.open(self.image_path))
        image_copy = np.array(Image.fromarray(self.black_white_mask).convert("RGB"))
        ##These lines temporarily added by MFQ just to plot paths on image-end
        h, w = image.shape[:2]
        path_image = np.ones((h, w, 3), dtype=np.uint8) * 255

        ##These lines temporarily added by MFQ just to plot paths on image-start
        # Customizable path color and thickness
        path_color = (255, 0, 0)  # Red color (B, G, R)
        path_thickness = 10  # Adjust as needed
        ##These lines temporarily added by MFQ just to plot paths on image-end

        # 1. Draw paths on the white image
        for path in self.paths:
            color = np.random.choice(range(256), size=3).tolist()
            for i in range(len(path) - 1):
                cv2.line(path_image, path[i], path[i + 1], color, 2)
                ##These lines temporarily added by MFQ just to plot paths on image-start
                cv2.line(image_copy, path[i], path[i + 1], path_color, path_thickness)
                ##These lines temporarily added by MFQ just to plot paths on image-end
        Image.fromarray(path_image).save(f"{self.output_folder}/path_image.png", "PNG")
        ##These lines temporarily added by MFQ just to plot paths on image-start
        # Save or display the modified original image
        Image.fromarray(image_copy).save(
            f"{self.output_folder}/original_with_paths.png", "PNG"
        )
        ##These lines temporarily added by MFQ just to plot paths on image-end

        # 2. Find contours
        gray_image = cv2.cvtColor(path_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print("This is length contour", len(contours))

        # 3 & 4. Create masks and extract subimages
        for i, contour in enumerate(contours):
            mask = np.zeros_like(gray_image)
            cv2.drawContours(
                image=mask,
                contours=[contour],
                contourIdx=-1,
                color=WHITE_BW_COLOR,
                thickness=cv2.FILLED,
            )  # Fill the contour
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2RGBA)
            masked_image[:, :, 3] = mask

            # Extract the bounding rectangle of the masked region
            x, y, w, h = cv2.boundingRect(contour)
            subimage = masked_image[y : y + h, x : x + w]

            if self.superRes:
                Image.fromarray(subimage).save(
                    f"{self.subimages_before_super_resolution_folder}/{self.image_name}_subimage_{i}.png",
                )
                subimage = self.sr.superize(subimage)
            elif self.add_background:
                subimage = np.array(self.create_image_with_background(subimage))

            if subimage.shape[0] > 10 and subimage.shape[1] > 10:
                Image.fromarray(subimage).save(
                    f"{self.subimages_folder}/{self.image_name}_subimage_{i}.png",
                )
            else:
                logger.info("Sub-image is less than 10 pixel length or width")

    def create_image_with_background(self, foreground):
        height, width, _ = foreground.shape
        if width >= self.pad_output_width and height >= self.pad_output_height:
            return foreground

        output = Image.new(
            "RGBA",
            (self.pad_output_width, self.pad_output_height),
            MILLER_PINK_RGBA_COLOR,
        )
        output.paste(
            im=Image.fromarray(foreground),
            box=(
                (self.pad_output_width - width) // 2,
                (self.pad_output_height - height) // 2,
            ),
            mask=Image.fromarray(foreground),
        )
        return np.array(output)

    def get_number_of_masks(self):
        number_of_black_color = 1
        return (
            len(np.unique(np.array(self.mask_image).reshape(-1, 4), axis=0))
            - number_of_black_color
        )

    def process(self):
        """Main processing function to execute all steps."""
        n_masks = self.get_number_of_masks()
        if n_masks < self.min_number_of_masks_to_divide_image:
            logger.info(
                "Skipping image division due to mask low count, actual count %s",
                n_masks,
            )
        else:
            self.connect_points_with_paths()
        self.split_image_by_paths()


def draw_non_overlapping_bounding_boxes(
    draw, bounding_boxes, labels, min_width=2, max_width=5, text_offset=2, font_size=30
):
    # Use a system font on Linux
    font = (
        ImageFont.load_default()
        if font_size <= 0
        else ImageFont.truetype("DejaVuSans.ttf", font_size)
    )

    occupied_regions = []
    colors = [
        "#ff1493",
        "#1e90ff",
        "#8b4513",
        "#ffd700",
        "#00ced1",
        "#ff4500",
        "#4b0082",
    ]
    for color, label, bbox in zip(colors, labels, bounding_boxes):
        min_x, min_y, max_x, max_y = bbox
        line_width = random.randint(min_width, max_width)
        draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=line_width)
        # --- Label Placement and Overlap Prevention ---
        text_x = min_x + text_offset
        text_y = min_y + text_offset
        text_w, text_h = font.getbbox(label)[2:]
        # Check for overlap with previous labels
        while any(
            text_x < region[2]
            and text_x + text_w > region[0]
            and text_y < region[3]
            and text_y + text_h > region[1]
            for region in occupied_regions
        ):
            text_y += text_h / 4  # Move label down if overlap detected
        # Add the current label's region to occupied_regions
        occupied_regions.append((text_x, text_y, text_x + text_w, text_y + text_h))
        # Draw the label (using box color as background)
        draw.rectangle([text_x, text_y, text_x + text_w, text_y + text_h], fill=color)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)


def get_clustered_bounding_boxes(bboxes, k=7):
    """
    Draws k bounding boxes representing merged clusters of original boxes with labels.
    Clusters are formed based on the proximity of bounding box centers.
    Prevents label overlap and matches label background to box color.
    """
    # Cluster the bounding boxes using KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(bboxes)
    cluster_labels = kmeans.labels_

    # Iterate through each cluster
    clustered_bboxes = list()
    for cluster_id in range(k):
        cluster_bboxes = bboxes[cluster_labels == cluster_id]
        # Find the min/max coordinates of boxes in the cluster
        min_x = int(cluster_bboxes[:, 0].min())
        min_y = int(cluster_bboxes[:, 1].min())
        max_x = int(cluster_bboxes[:, 2].max())
        max_y = int(cluster_bboxes[:, 3].max())
        clustered_bboxes.append(np.array([min_x, min_y, max_x, max_y]))

    return np.array(clustered_bboxes)


def merge_bounding_boxes(boxes):
    def intersect(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return x1 < x2 and y1 < y2

    boxes = sorted(
        boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
    )

    merge_happened = True
    while merge_happened:
        i = 0
        current = boxes.pop(0)
        merge_happened = False
        while i < len(boxes):
            if intersect(current, boxes[i]):
                x1 = min(current[0], boxes[i][0])
                y1 = min(current[1], boxes[i][1])
                x2 = max(current[2], boxes[i][2])
                y2 = max(current[3], boxes[i][3])
                current = [x1, y1, x2, y2]
                boxes.pop(i)
                merge_happened = True
            else:
                i += 1
        boxes.append(current)
    return boxes


def expand_box(box, image_width, image_height, factor=1.5):
    width = box[2] - box[0]
    height = box[3] - box[1]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    new_width = width * factor
    new_height = height * factor

    # Calculate new coordinates
    x1 = center_x - new_width / 2
    y1 = center_y - new_height / 2
    x2 = center_x + new_width / 2
    y2 = center_y + new_height / 2

    # Clamp coordinates to image boundaries
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return [x1, y1, x2, y2]


def negative_prompter(question, image_pil):
    PROMPT = 'What area/object from the image is not asked about in "{question}".\n\nThe output should be brief, starting with "The question does not asked about "'  # noqa: E501
    client = OpenAI(api_key=OPENAI_API_KEY)

    buffer = BytesIO()
    image_pil.save(buffer, "PNG")
    response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                            },
                        },
                        {
                            "type": "text",
                            "text": PROMPT.format(question=question),
                        },
                    ],
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "text"},
        )
        .choices[0]
        .message.content
    )
    logger.info(f"chatgpt excluded: {response}")
    formatted_response = response.replace("The question does not asked about", "")
    logger.info(f"chatgpt formatted excluded: {formatted_response}")
    return formatted_response


def llm_bounding_box_selector(object_of_interest, negative_prompt, image_pil):
    PROMPT = "Which bounding boxes contain all the {object_of_interest}. Note that the bounding boxes you give should exclude {negative_prompt}."
    client = OpenAI(api_key=OPENAI_API_KEY)

    buffer = BytesIO()
    image_pil.save(buffer, "PNG")

    response = json.loads(
        client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                            },
                        },
                        {
                            "type": "text",
                            "text": PROMPT.format(object_of_interest=object_of_interest, negative_prompt=negative_prompt),
                        },
                    ],
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bounding_box_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "bounding_box_list": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "additionalProperties": False,
                        "required": ["bounding_box_list"],
                    },
                },
            },
        )
        .choices[0]
        .message.content
    )
    logger.info(f"chatgpt areas selected: {response}")
    return response["bounding_box_list"]


def chatgpt_crop_image_from_bounding_boxes(
    image_path, bounding_boxes, question, object_of_interest, output_folder
):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    crops_folder = os.path.join(output_folder, f"{image_name}_crops")
    os.makedirs(crops_folder, exist_ok=True)

    if len(bounding_boxes) == 0:
        logger.info(f"As the number of bbox is ZERO, returning the whole image, didn't use ChatGPT.")
        image = Image.open(image_path)
        image.save(os.path.join(crops_folder, f"{image_name}_crop_1.png"))
    if len(bounding_boxes) == 1:
        logger.info(f"As the number of bbox is ONE, returning just that area, didn't use ChatGPT.")

        image = Image.open(image_path)
        image_width, image_height = image.size
        bbox = bounding_boxes.numpy()[0]
        bbox = expand_box(bbox, image_width, image_height)
        x_min, y_min, x_max, y_max = bbox
        cropped_image = F.crop(image, y_min, x_min, y_max - y_min, x_max - x_min)
        cropped_image.save(os.path.join(crops_folder, f"{image_name}_crop_1.png"))

        draw = ImageDraw.Draw(image)
        draw_non_overlapping_bounding_boxes(
            draw, bounding_boxes, [str(i + 1) for i in range(len(bounding_boxes))]
        )
        image.save(f"{crops_folder}/clustered_bbox.png")
    else:
        negative_prompt = negative_prompter(
            question=question, image_pil=Image.open(image_path)
        )

        bounding_boxes = bounding_boxes.numpy() # get_clustered_bounding_boxes(bounding_boxes.numpy())
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        draw_non_overlapping_bounding_boxes(
            draw, bounding_boxes, [str(i + 1) for i in range(len(bounding_boxes))]
        )
        image.save(f"{crops_folder}/clustered_bbox.png")

        llm_selected_bboxes = llm_bounding_box_selector(
            object_of_interest, negative_prompt, image
        )

        selected_bboxes = list()
        for bbox_number in llm_selected_bboxes:
            try:
                bbox_index = int(bbox_number) - 1
                if bbox_index > 0:
                    selected_bboxes.append(bounding_boxes[bbox_index])
            except ValueError:
                logger.warning(f"Given bounding box '{bbox_number}' could not cast to int", exc_info=True)
            except IndexError:
                logger.warning(f"Given bounding box number '{bbox_number}' does not exist", exc_info=True)

        if len(selected_bboxes) > 0:
            image = Image.open(image_path)
            image_width, image_height = image.size
            expanded_bboxes = [
                expand_box(box, image_width, image_height) for box in selected_bboxes
            ]

            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            draw_non_overlapping_bounding_boxes(
                draw,
                [expand_box(box, image_width, image_height) for box in bounding_boxes],
                [str(i + 1) for i in range(len(bounding_boxes))],
                min_width=5,
                max_width=5,
            )
            image.save(f"{crops_folder}/expaned_bboxes.png")

            merged_bboxes = merge_bounding_boxes(expanded_bboxes)
            cropped_images = []
            image = Image.open(image_path)
            for box in merged_bboxes:
                x_min, y_min, x_max, y_max = box
                cropped_image = F.crop(image, y_min, x_min, y_max - y_min, x_max - x_min)
                cropped_images.append(cropped_image)

            for i, cropped_image in enumerate(cropped_images):
                cropped_image.save(os.path.join(crops_folder, f"{image_name}_crop_{i+1}.png"))
        elif len(bounding_boxes) > 0:
            crop_image_from_bounding_boxes(
                image_path=image_path, 
                bounding_boxes=torch.from_numpy(bounding_boxes), 
                output_folder=output_folder 
            )
        else:
            cropped_image = Image.open(image_path)
            cropped_image.save(os.path.join(crops_folder, f"{image_name}_crop_1.png"))


def crop_image_from_bounding_boxes(
    image_path, bounding_boxes, output_folder, sam_mask_image=None, sam_mask=None
):
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    crops_folder = os.path.join(output_folder, f"{image_name}_crops")
    os.makedirs(crops_folder, exist_ok=True)
    bounding_boxes = bounding_boxes.numpy()  # Convert tensor to NumPy array

    # remain_boxes, _ = process_boxes(image, bounding_boxes) # To resolve the fsc-147 dataset abundant object problem, we want not to remove the largest covering boxes
    remain_boxes = bounding_boxes
    # Merge overlapping bounding boxes
    merged_boxes = merge_boxes(remain_boxes)

    _, ax = plt.subplots()
    im = Image.open(image_path)
    ax.imshow(im)
    for i, box in enumerate(merged_boxes):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.savefig(os.path.join(crops_folder, f"gdino_area_{image_name}_{i+1}.png"))
    plt.close()

    # if you want mdetr to intersect uncomment, if not intersected_boxes = merged_boxes
    # intersected_boxes = compare_boxes(merged_boxes, re_mbr_box)
    intersected_boxes = merged_boxes

    cropped_images = []
    for box in intersected_boxes:
        x_min, y_min, x_max, y_max = box
        # Ensure bounding box coordinates are integers
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cropped_image = F.crop(image, y_min, x_min, y_max - y_min, x_max - x_min)
        cropped_images.append(cropped_image)

    # Ensure output folder and subfolder exist
    # Do something with the cropped images (e.g., display, save)
    for i, cropped_image in enumerate(cropped_images):
        cropped_image.save(os.path.join(crops_folder, f"{image_name}_crop_{i+1}.png"))

    if (sam_mask_image is not None) and (sam_mask is not None):
        sam_cropped_images = []
        for box in intersected_boxes:
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            sam_cropped_image = F.crop(sam_mask_image, y_min, x_min, y_max - y_min, x_max - x_min)
            sam_cropped_images.append(sam_cropped_image)

        for i, sam_cropped_image in enumerate(sam_cropped_images):
            sam_cropped_image.save(os.path.join(crops_folder, f"{image_name}_sam_{i+1}.png"))

        filtered_sam_mask = list()
        for box in intersected_boxes:
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cropped_sam_mask = sam_mask[:, y_min:y_max, x_min:x_max]

            filtered_cropped_sam_mask = list()
            for m in cropped_sam_mask:
                if m.sum() > 0:
                    filtered_cropped_sam_mask.append(m)
            filtered_sam_mask.append(np.array(filtered_cropped_sam_mask))
        return filtered_sam_mask
    else:
        return None

def trace_contour(edges, start_x, start_y):
    contour = []
    x, y = start_x, start_y

    while True:
        contour.append((x, y))
        edges[y, x] = False

        # Check 8-connected neighbors
        for dx, dy in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < edges.shape[0] and 0 <= nx < edges.shape[1] and edges[ny, nx]:
                x, y = nx, ny
                break
        else:
            break  # No more connected pixels

    return contour


# code for merge boxes
def merge_boxes(boxes):
    # Sort boxes based on area (smallest to largest)
    boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def intersect(box1, box2):
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        return x1_max < x2_min and y1_max < y2_min

    def merge(box1, box2):
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]

    merged = False
    while not merged:
        merged = True
        for i in range(len(boxes) - 1):
            for j in range(i + 1, len(boxes)):
                if intersect(boxes[i], boxes[j]):
                    boxes[i] = merge(boxes[i], boxes[j])
                    del boxes[j]
                    merged = False
                    break
            if not merged:
                break
        boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    return boxes


def calculate_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_mbr(boxes):
    if len(boxes) == 0:
        return None
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    return np.array([x_min, y_min, x_max, y_max])


def process_boxes(image, bounding_boxes):
    box_list = sorted(bounding_boxes, key=calculate_area, reverse=True)
    covering_boxes = []

    while len(box_list) > 1:
        largest_box = box_list[0]
        rest_boxes = box_list[1:]

        mbr = calculate_mbr(rest_boxes)
        mbr_area = calculate_area(mbr)
        
        largest_box_mbr_intersection = get_intersection(largest_box, mbr)
        if largest_box_mbr_intersection is None:
            break

        largest_box_mbr_intersection_area = calculate_area(largest_box_mbr_intersection)
        if largest_box_mbr_intersection_area / mbr_area > 0.95:
            covering_boxes.append(largest_box)
            box_list = rest_boxes
        else:
            break

    # # Draw the resulting boxes on the image
    # for box in box_list:
    #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # for box in covering_boxes:
    #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    # # Save the new image
    # cv2.imwrite(f"outputs/{image_name}_remove.jpg", image)

    return box_list, covering_boxes


# this function returns the intersection of the boxes
def get_intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2, y2], dtype=float)
    else:
        return None


# this function compres the boxes in the list
def compare_boxes(box_list, single_box):
    intersection_boxes = []

    for box in box_list:
        intersection = get_intersection(box, single_box)
        if intersection is not None:
            intersection_boxes.append(intersection)

    return intersection_boxes
