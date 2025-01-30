python3 eval_emoji.py \
    --dataset_path emoji_benchmark/benchmark_one_canvas/benchmark_data.json \
    --image_base_path emoji_benchmark/benchmark_one_canvas/images \
    --mask_detection_box_threshold 0.05 \
    --mask_detection_text_threshold 0.05 \
    --mask_detection_iou_threshold 0.8 \
    --area_detection_box_threshold 0.2 \
    --area_detection_text_threshold 0.2 \
    --area_detection_iou_threshold 0.8