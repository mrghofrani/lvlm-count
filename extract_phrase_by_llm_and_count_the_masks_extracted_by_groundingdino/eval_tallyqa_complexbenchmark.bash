python3 eval_tallyqa.py \
    --dataset_path data/tallyqa/benchmark_complex.json \
    --image_base_path data/genome \
    --mask_detection_box_threshold 0.1 \
    --mask_detection_text_threshold 0.1 \
    --mask_detection_iou_threshold 0.8 \
    --area_detection_box_threshold 0.15 \
    --area_detection_text_threshold 0.15 \
    --area_detection_iou_threshold 0.8 \
    --super_resolution