now=$(date +"%Y%m%d_%H%M%S")
config_file=config_resnetv1ce50_step_moving_average.yaml
python -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py  \
--config configs/$config_file 2>&1|tee log/train-$config_file-$now.log
