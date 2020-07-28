now=$(date +"%Y%m%d_%H%M%S")
config_file=configs/config_resnetv1ce50_step_moving_average.yaml
python -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py  \
--config $config_file -e #2>&1|tee test-$now.log




