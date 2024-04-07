# (step 1) download p2 weighting ffhq_p2.pt and run code

DEVICE=0

SAMPLE_FLAGS="--attention_resolutions 16 --class_cond False --class_cond False --diffusion_steps 1000 --dropout 0.0 \
    --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 \
    --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 50"

CUDA_VISIBLE_DEVICES=${DEVICE} \
python gen_style_domA.py ${SAMPLE_FLAGS} \
    --model_path P2_weighting/models/ffhq_p2.pt \
    --input_dir imgs_style_domB \
    --sample_dir imgs_style_domA \
    --img_name img1.png \
    --n 1 \
    --t_start_ratio 0.5 \
    --seed 1 \
