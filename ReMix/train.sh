exp_name='EXP' #exp name
num_classes=4
dataset='Camelyon16' # dataset name
in_dim=1280
sag='hg tg'
hg_path='' # HG_PATH
hg_loss='mse'
tg_path='' # TG_PATH
tg_loss='in-out'
weighting='UW'


python train_sag.py --gpu_index 0 --exp_name $exp_name --seed $RANDOM \
--num_classes $num_classes --dataset $dataset --feats_size $in_dim \
--sag $sag --hg_dir $hg_path --hg_loss $hg_loss --tg_dir $tg_path --tg_loss $tg_loss \
--weighting $weighting

# python test.py --gpu_index 0 --num_classes $num_classes --dataset $dataset --feats_size $in_dim --multiple-dir --checkpoint-dir $checkpoint_dir