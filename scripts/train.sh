# download pretrained DiffAE
# batch size 8 -> 34GB / single A100 about 30min
# (step 2)

DEVICE=0

CUDA_VISIBLE_DEVICES=${DEVICE} \
python train_diffaeB.py \
    --style_domA_dir imgs_style_domA \
    --style_domB_dir imgs_style_domB \
    --ref_img img1.png \
    --work_dir exp/img1 \
    --n_iter 200 \
    --ckpt_freq 200 \
    --batch_size 8 \
    --map_net \
    --map_time \
    --lambda_map 0.1 \
    --train
