
# Base feature extractor
base_extractor="mobilenetv2"
weights="model/pretrained_cnn_models/mobilenetv2_s_1.0_imagenet_224x224.pth" # feature extractor weights

# dataset
data="sample_data" # path to the dataset txt files
num_classes=4
resize1_scale="0 1 2" # chosen scale index
patch_num="5 7 9" # number of patches (each side) for different scale inputs

# attention setting
attn="HG_PATH"
attn_loss="mse"
tissue_attn="TG_PATH"

# model setting
output_dir="./runs" # directory to save model
channels=3
model_dim=128 # linear projection dimension
n_layers=4
head_dim=32
drop_out=0.2
in_dim=1280 # input embedding dimension
linear_channel=4
num_scale_attn_layer=2
use_standard_emb=True

# general setting
workers=4
batch_size=16
gpu_id="0 1"
mode="train"
savedir="./"
epochs=200 
model="multi_resolution"
seed=$RANDOM

# optimizer setting
aggregate_batch=1
weighting="UW"
optim="adam"
loss_function="bce"
weight_decay=4e-05

# scheduler setting
lr=0.0005
lr_decay=0.5
patience=50
scheduler="step"

CUDA_VISIBLE_DEVICES=0,1 python main.py --model-dir $output_dir --savedir $output_dir \
--data $data --num-classes $num_classes --binarize --resize1-scale $resize1_scale --patch-num $patch_num \
--attn-guide --tissue-constraint --attn $attn --attn-loss $attn_loss --tissue-attn $tissue_attn \
--in-dim $in_dim --workers $workers --batch-size $batch_size --loss-function $loss_function --optim $optim \
--use-gpu --gpu-id $gpu_id --use-parallel --mode $mode --epochs $epochs --seed $seed --model $model \
--weighting $weighting

# CUDA_VISIBLE_DEVICES=0,1 python model_select.py --checkpoint-dir $output_dir --multiple-dir --load-config --mode test-on-train-valid