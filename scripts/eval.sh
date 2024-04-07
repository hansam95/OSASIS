# (step 3) Evaluate

DEVICE=0

CUDA_VISIBLE_DEVICES=${DEVICE} \
python eval_diffaeB.py \
    --style_domB_dir imgs_style_domB \
    --infer_dir imgs_input_domA \
    --ref_img img1.png \
    --work_dir exp/img1 \
    --map_net \
    --map_time \
    --lambda_map 0.1