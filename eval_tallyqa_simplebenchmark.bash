python3 eval_tallyqa.py \
    --dataset_path data/tallyqa/benchmark_simple.json \
    --image_base_path data/genome \
    --mask_detection_box_threshold 0.1 \
    --mask_detection_text_threshold 0.1 \
    --mask_detection_iou_threshold 0.8 \
    --area_detection_box_threshold 0.2 \
    --area_detection_text_threshold 0.2 \
    --area_detection_iou_threshold 0.8