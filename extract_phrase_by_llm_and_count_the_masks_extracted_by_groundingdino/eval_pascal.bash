python3 eval_pascal.py \
    --dataset_path data/pascal/sampled_pascal.csv \
    --image_base_path data/PASCAL/VOCdevkit/VOC2007/JPEGImages \
    --mask_detection_box_threshold 0.1 \
    --mask_detection_text_threshold 0.1 \
    --mask_detection_iou_threshold 0.8 \
    --area_detection_box_threshold 0.15 \
    --area_detection_text_threshold 0.15 \
    --area_detection_iou_threshold 0.15
