import logging
import os
import random
import sys

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamPredictor, build_sam

WHITE_RGBA_COLOR = (255, 255, 255, 255)
TRANSPARENT_BLACK_RGBA_COLOR = (0, 0, 0, 0)

TRANSPARENT_BOUNDARY_BORDER_WIDTH = 0


logger = logging.getLogger("logger")


class GroundedSAM:
    def __init__(self, config_file, ckpt_filename, sam_checkpoint,
                 factor_to_detect_abnormal_large_masks, device=None):
        self.config_file = config_file
        self.ckpt_filename = ckpt_filename
        self.sam_checkpoint = sam_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_to_detect_abnormal_large_masks = factor_to_detect_abnormal_large_masks

        self.groundingdino_model = None
        self.sam_predictor = None

        sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
        sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

    def load_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def transform_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)
        return image

    def get_grounding_output(
        self, model, image, caption, box_threshold, text_threshold, with_logits=True
    ):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        scores = []

        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def draw_mask(self, mask, draw, random_color=False):
        color = (
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                153,
            )
            if random_color
            else (30, 144, 255, 153)
        )
        nonzero_coords = np.transpose(np.nonzero(mask))
        for coord in nonzero_coords:
            draw.point(coord[::-1], fill=color)

    def draw_box(self, box, draw, label):
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)
        if label:
            font = ImageFont.load_default()
            bbox = (
                draw.textbbox((box[0], box[1]), str(label), font)
                if hasattr(font, "getbbox")
                else (box[0], box[1], *draw.textsize(str(label), font))
            )
            draw.rectangle(bbox, fill=color)
            draw.text((box[0], box[1]), str(label), fill="white")
            draw.text((box[0], box[1]), label)

    def run_grounded_sam(
            self,
            image_path,
            text_prompt,
            task_type,
            box_threshold,
            text_threshold,
            iou_threshold,
            sam_masks=None
    ):
        input_image = Image.open(image_path)
        image_pil = input_image.convert("RGB")
        transformed_image = self.transform_image(image_pil)

        if self.groundingdino_model is None:
            self.groundingdino_model = self.load_model(
                self.config_file, self.ckpt_filename
            )
        size = image_pil.size
        H, W = size[1], size[0]

        try:
            boxes_filt, scores, pred_phrases = self.get_grounding_output(
                self.groundingdino_model,
                transformed_image,
                text_prompt,
                box_threshold,
                text_threshold,
            )
        except RuntimeError: # When the area selected is soo small (140*2) selected is soo small an error is generated.
            logger.exception("The area selected is soo small.", exc_info=True)
            mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))

            image_pil = image_pil.convert("RGBA")
            image_pil.alpha_composite(mask_image)
            return [image_pil, mask_image]

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()

        if task_type == "det":
            return [image_pil, boxes_filt]

        elif task_type == "seg":
            if self.sam_predictor is None:
                sam = build_sam(checkpoint=self.sam_checkpoint)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)

            image = np.array(image_pil)
            self.sam_predictor.set_image(image)

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_filt, image.shape[:2]
            ).to(self.device)
            try:
                masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            except RuntimeError: # When GroundingDino could not detect any masks raises an error. Note: SAM masks still could be used, but I didn't used because this error happens when the image size is small.
                logger.exception("Image size is soo small, no mask is detected.", exc_info=True)
                mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))

                image_pil = image_pil.convert("RGBA")
                image_pil.alpha_composite(mask_image)
                return [image_pil, mask_image]

            boxes_filt = boxes_filt.numpy()
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()

            masks, nms_idx = self.suppress_redundant_masks(boxes_filt, scores, masks)
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]

            unfiltered_ids = self.filter_abnormal_mask(masks)
            boxes_filt = boxes_filt[unfiltered_ids]
            masks = masks[unfiltered_ids]
            pred_phrases = [pred_phrases[idx] for idx in unfiltered_ids]
            masks = masks.squeeze(axis=1)

            logger.info("number of gdino_masks: %s", len(masks))
            if sam_masks is not None and \
                    len(sam_masks)/len(masks) > 6:
                logger.info("Using SAM masks.")
                unfiltered_ids = self.filter_abnormal_mask(sam_masks)
                masks = sam_masks[unfiltered_ids]

            masks = self.erode_masks(masks)
            masks = self.polish_masks(masks)
            masks = self.remove_1pixel_border(masks)

            mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            for mask in masks:
                self.draw_mask(mask, mask_draw, random_color=True)
            image_draw = ImageDraw.Draw(image_pil)
            for box, label in zip(boxes_filt, pred_phrases):
                self.draw_box(box, image_draw, label)
            image_pil = image_pil.convert("RGBA")
            image_pil.alpha_composite(mask_image)

            mask_image = self.add_transparent_boundaries(mask_image)
            return [image_pil, mask_image]

    def add_transparent_boundaries(self, mask_image):
        mask_array = np.array(mask_image)

        output = np.zeros((mask_array.shape), dtype=np.uint8)
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGBA2GRAY)
        _, mask_black_white_array = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask_black_white_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(
            image=output,
            contours=contours,
            contourIdx=-1,
            color=WHITE_RGBA_COLOR,
            thickness=cv2.FILLED,
        )
        cv2.drawContours(
            image=output,
            contours=contours,
            contourIdx=-1,
            color=TRANSPARENT_BLACK_RGBA_COLOR,
            thickness=TRANSPARENT_BOUNDARY_BORDER_WIDTH,
        )

        return mask_image  # Image.fromarray(output * np.array(mask_image))

    def polish_masks(self, masks):
        def clean_mask(mask):
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1
            )
            return cleaned_mask

        return np.array([clean_mask(mask.astype(np.uint8)) for mask in masks])

    def filter_abnormal_mask(self, masks):
        def get_indices_of_zero_masks(masks):
            return [i for i, mask in enumerate(masks) if mask.sum() == 0]

        def get_indices_of_abnormally_large_masks(masks, ignore_indices):
            median_mask_area = np.median([mask.sum() for i, mask in enumerate(masks) if i not in ignore_indices])
            indices_of_large_masks = []
            for i in range(len(masks)):
                if masks[i].sum() > median_mask_area * self.factor_to_detect_abnormal_large_masks:
                    indices_of_large_masks.append(i)
            return indices_of_large_masks

        zero_mask_indices = get_indices_of_zero_masks(masks)
        abnormally_large_mask_indices = get_indices_of_abnormally_large_masks(masks, ignore_indices=zero_mask_indices)

        indices_to_be_deleted = zero_mask_indices + abnormally_large_mask_indices
        return [i for i in range(len(masks)) if i not in indices_to_be_deleted]

    def calculate_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def calculate_mbr(self, bboxes):
        x_min = np.min(bboxes[:, 0])
        y_min = np.min(bboxes[:, 1])
        x_max = np.max(bboxes[:, 2])
        y_max = np.max(bboxes[:, 3])
        return [x_min, y_min, x_max, y_max]

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = self.calculate_area(box1)
        area2 = self.calculate_area(box2)

        iou = intersection / (area1 + area2 - intersection)
        conver_ratio = intersection / area2
        return iou, conver_ratio

    def filter_bboxes(self, bboxes, scores, masks):
        areas = np.array([self.calculate_area(bbox) for bbox in bboxes])
        sorted_indices = np.argsort(areas)[::-1]

        filtered_indices = []
        for i in range(len(sorted_indices)):
            current_bbox = bboxes[sorted_indices[i]]
            remaining_bboxes = bboxes[sorted_indices[i+1:]]

            if len(remaining_bboxes) == 0:
                filtered_indices.append(sorted_indices[i])
                break

            mbr = self.calculate_mbr(remaining_bboxes)
            iou, cover_ratio = self.calculate_iou(current_bbox, mbr)
            if cover_ratio <= 0.67:
                filtered_indices.extend(sorted_indices[i:])
                break

        return bboxes[filtered_indices], scores[filtered_indices], masks[filtered_indices], filtered_indices

    def mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union

    def nms_masks(self, masks, scores, iou_threshold=0.4):
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            ious = np.array([self.mask_iou(masks[i, 0], masks[j, 0]) for j in order[1:]])
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def suppress_redundant_masks(self, bboxes, scores, masks):
        filtered_bboxes, filtered_scores, filtered_masks, filtered_indices = self.filter_bboxes(bboxes, scores, masks)
        keep_indices = self.nms_masks(filtered_masks, filtered_scores)
        final_masks = filtered_masks[keep_indices]
        final_masks = self.rectify_intersection(final_masks)
        nms_indices = np.array(filtered_indices)[keep_indices]
        return final_masks, nms_indices

    def rectify_intersection(self, masks):
        m, _, height, width = masks.shape
        # Flatten the masks for easier comparison
        flat_masks = masks.reshape(m, height, width)

        for i in range(m):
            for j in range(i + 1, m):
                mask_i = flat_masks[i]
                mask_j = flat_masks[j]

                # Find intersection
                intersection = mask_i & mask_j

                if np.any(intersection):
                    # Calculate the area of the masks
                    area_i = np.sum(mask_i)
                    area_j = np.sum(mask_j)

                    # Remove intersection area from the smaller mask
                    if area_i < area_j:
                        flat_masks[i] = mask_i & ~intersection
                    else:
                        flat_masks[j] = mask_j & ~intersection

        # Reshape the masks back to original shape
        return flat_masks.reshape(m, 1, height, width)

    def erode_masks(self, masks):
        kernel = np.ones((2, 2), np.uint8)
        eroded_masks = np.zeros_like(masks, dtype=np.uint8)  # Specify dtype here
        for i in range(masks.shape[0]):
            eroded_masks[i] = cv2.erode(masks[i].astype(np.uint8), kernel)  # Convert to uint8

        eroded_masks = eroded_masks.astype(bool)
        return eroded_masks

    def remove_1pixel_border(self, masks):
        masks[:, 0, :] = 0  # Top border
        masks[:, -1, :] = 0  # Bottom border
        masks[:, :, 0] = 0  # Left border
        masks[:, :, -1] = 0  # Right border
        return masks
