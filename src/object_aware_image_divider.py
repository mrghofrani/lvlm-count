import logging
from src.grounded_sam import GroundedSAM
from src.image_divider import ImageDivider, crop_image_from_bounding_boxes
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import random
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger("logger")


class ObjectAwareImageDivider:
    GROUDING_DINO_CONFIG_FILE = (
        "src/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    )
    GROUNDING_DINO_CHECKPOINT_FILE = "weights/groundingdino_swinb_cogcoor.pth"
    SAM_CHECKPOINT_FILE = "weights/sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "default"

    def __init__(
        self,
        mask_detection_box_threshold,
        mask_detection_text_threshold,
        mask_detection_iou_threshold,
        area_detection_box_threshold,
        area_detection_text_threshold,
        area_detection_iou_threshold,
        use_sam_masks=False,
        factor_to_detect_abnormal_large_masks=25,
        device=None
    ):
        self.use_sam_masks = use_sam_masks
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_detection_box_threshold = mask_detection_box_threshold
        self.mask_detection_text_threshold = mask_detection_text_threshold
        self.mask_detection_iou_threshold = mask_detection_iou_threshold
        self.area_detection_box_threshold = area_detection_box_threshold
        self.area_detection_text_threshold = area_detection_text_threshold
        self.area_detection_iou_threshold = area_detection_iou_threshold

        if self.use_sam_masks:
            self.sam = sam_model_registry[self.SAM_MODEL_TYPE](checkpoint=self.SAM_CHECKPOINT_FILE)
            self.sam = self.sam.to(device=self.device)
            self.sam_mask_generator_model = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
            )

        self.grounded_sam = GroundedSAM(
            config_file=self.GROUDING_DINO_CONFIG_FILE,
            ckpt_filename=self.GROUNDING_DINO_CHECKPOINT_FILE,
            sam_checkpoint=self.SAM_CHECKPOINT_FILE,
            factor_to_detect_abnormal_large_masks=factor_to_detect_abnormal_large_masks
        )

        self.sam_masks = None
        self.sam_mask_image = None

    def divide_by_prompt(
            self,
            image_path,
            prompt,
            output_folder,
            vertical_divides,
            horizontal_divides,
            equi_size_div=False,
    ):
        if self.use_sam_masks:
            crop_index = int(image_path.split("/")[-1].split(".")[0].split("_")[-1]) - 1
            logger.info("number of sam_masks: %s", len(self.sam_masks[crop_index]))
            bbox_image, mask_image = self.grounded_sam.run_grounded_sam(
                image_path=image_path,
                text_prompt=prompt,
                task_type="seg",
                box_threshold=self.mask_detection_box_threshold,
                text_threshold=self.mask_detection_text_threshold,
                iou_threshold=self.mask_detection_iou_threshold,
                sam_masks=self.sam_masks[crop_index]
            )
        else:
            bbox_image, mask_image = self.grounded_sam.run_grounded_sam(
                image_path=image_path,
                text_prompt=prompt,
                task_type="seg",
                box_threshold=self.mask_detection_box_threshold,
                text_threshold=self.mask_detection_text_threshold,
                iou_threshold=self.mask_detection_iou_threshold
            )

        bbox_image.save(f"{output_folder}/bbox.png", "PNG")
        mask_image.save(f"{output_folder}/mask.png", "PNG")

        divider = ImageDivider(
            image_path=image_path,
            mask_image=mask_image,
            output_folder=output_folder,
            vertical_divides=vertical_divides,
            horizontal_divides=horizontal_divides,
            equi_size_div=equi_size_div,
        )
        divider.process()

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

    def sam_mask_generator(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        image = np.array(image_pil)
        sam_masks = np.array(
            [
                mask["segmentation"].astype(np.uint8)
                for mask in self.sam_mask_generator_model.generate(image)
            ]
        )
        sam_mask_image = Image.new("RGBA", image_pil.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(sam_mask_image)
        for mask in sam_masks:
            self.draw_mask(mask, mask_draw, random_color=True)
        return sam_masks, sam_mask_image

    def detect_area(
            self,
            image_path,
            gdino_prompt,
            output_folder,
            question=None,
            object_of_interest=None,
            chatgpt_enabled=False
    ):
        _, bounding_boxes = self.grounded_sam.run_grounded_sam(
            image_path=image_path,
            text_prompt=gdino_prompt,
            task_type="det",
            box_threshold=self.area_detection_box_threshold,
            text_threshold=self.area_detection_text_threshold,
            iou_threshold=self.area_detection_iou_threshold,
        )

        if self.use_sam_masks:
            self.sam_masks, self.sam_mask_image = self.sam_mask_generator(image_path)
            self.sam_mask_image.save(f"{output_folder}/sam_mask.png")

        if chatgpt_enabled:
            chatgpt_crop_image_from_bounding_boxes(
                image_path, bounding_boxes, question, object_of_interest, output_folder
            )
        elif self.use_sam_masks:
            self.sam_masks = crop_image_from_bounding_boxes(image_path, bounding_boxes, output_folder,
                                                            sam_mask_image=self.sam_mask_image, sam_mask=self.sam_masks)
        else:
            crop_image_from_bounding_boxes(
                image_path, bounding_boxes, output_folder, sam_mask_image=self.sam_mask_image,
                sam_mask=self.sam_masks
            )