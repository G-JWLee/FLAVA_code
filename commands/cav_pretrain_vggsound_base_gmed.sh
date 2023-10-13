#!/bin/bash

export home_path="/base_path"
export home_data_path="/dataset_path"

if [ "$2" == "" ]; then
  echo "CUDA_VISIBLE_DEVICES is 0,1,2,3 by default"
  gpus="0,1,2,3"
else
  echo "CUDA_VISIBLE_DEVICES is $2"
  gpus=$2
fi

if [ "$3" == "" ]; then
  echo "Random seed is not specified, default is 2021"
  random_seed=2021
else
  echo "$3 is given as random seed"
  random_seed=$3
fi

if [ "$4" == "" ]; then
  echo "Default memory size is 2000"
  memory_size=2000
else
  echo "$4 is given as memory size"
  memory_size=$4
fi

if [ "$5" == "" ]; then
  echo "Default gradient stride is 0.1"
  grad_stride=0.1
else
  echo "$5 is given as gradient stride"
  grad_stride=$5
fi

if [ "$6" == "" ]; then
  echo "Default regularization strength is 0.05"
  reg_strength=0.05
else
  echo "$6 is given as regularization strength"
  reg_strength=$6
fi

export CUDA_VISIBLE_DEVICES=${gpus}
PYTHONPATH=. python main.py --config-name=cav_pretrain_vggsound -m \
environment.port=21994 \
backbone=cav \
backbone.args.mask_ratio_a=0.8 \
backbone.args.mask_ratio_v=0.8 \
criterion=fa_mae_cont \
criterion.args.norm_pix_loss=True \
criterion.args.load_local_path=${home_path}/FLAVA_code/baseline_ckpt/audioset_pretrained.pth \
criterion.args.get_va_recall_metric=True \
criterion.args.contrast_loss_weight=0.1 \
cl_algo=gmed \
cl_algo.args.grad_stride=${grad_stride} \
cl_algo.args.reg_strength=${reg_strength} \
cl_algo.args.mem_args.memory_size=${memory_size} \
data_augm=cav_augm \
data.target_task=['sports','music','vehicle','people','animals','home_nature','others_part1','others_part2'] \
data.skip_task=[] \
data.args.video_duration=4. \
data.args.audio_duration=10. \
data.args.use_audio=True \
data.args.num_frames=4 \
logging.eval_freq=2 \
logging.name=cav_base_vggsound_pretrain \
logging.suffix=_gmed_${random_seed}_${memory_size} \
logging.save_freq_mints=120 \
logging.print_freq=20 \
logging.save_freq=10 \
environment.seed=${random_seed} \
environment.workers=32 \
environment.slurm=False \
environment.world_size=1 \
environment.ngpu=4 \
environment.multiprocessing_distributed=True \
environment.distributed=True \
environment.dist_url=env:// \
environment.rank=-1 \
optim=adam \
optim.args.lr=1e-4 \
optim.args.betas=[0.95,0.999] \
optim.args.weight_decay=5e-7 \
optim.epochs=10 \
optim.batch_size=48 \
optim.per_gpu_batchsize=12 \
optim.layer_decay=1.0 \
optim.use_lr_scheduler=False \

