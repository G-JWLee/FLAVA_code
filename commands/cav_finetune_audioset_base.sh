#!/bin/bash

export home_path="/base_path"
export home_data_path="/dataset_path"

if [ "$2" == "" ]; then
  echo "No pretrained model path is specified"
  pretrain_dataset=""
else
  echo "Pretrained on $2"
  pretrain_dataset="$2"
fi

if [ "$3" == "" ]; then
  echo "No pretrained model path is specified"
  pretrain_path=""
else
  echo "Running finetuning on $3"
  pretrain_path="$3"
fi

if [ "$4" == "" ]; then
  echo "No trial specified, default is 1"
  trial=1
else
  echo "$4-th trial finetuning"
  trial=$4
fi

if [ "$5" == "" ]; then
  echo "Random seed is not specified, default is 2023"
  random_seed=2023
else
  echo "$5 is given as random seed"
  random_seed=$5
fi

if [ "$6" == "" ]; then
  echo "CUDA_VISIBLE_DEVICES is 0,1,2,3 by default"
  gpus="0,1,2,3"
else
  echo "CUDA_VISIBLE_DEVICES is $6"
  gpus=$6
fi

if [ "$7" == "" ]; then
  echo "Default port is 21000"
  port=21000
else
  echo "$7 is given as port"
  port=$7
fi

export CUDA_VISIBLE_DEVICES=${gpus}
PYTHONPATH=. python main.py --config-name=cav_finetune_audioset -m \
environment.port=${port} \
train_algo.main_worker=run \
backbone=cav \
backbone.args.mid_fusion_depth=10 \
data_augm=cav_augm \
data_augm.audio_data.args.freqm=48 \
data_augm.audio_data.args.timem=192 \
data_augm.audio_data.args.noise=True \
criterion=va_cls \
criterion.args.load_local_path=${home_path}/FLAVA_code/experiments/checkpoints/cav_base_audioset_pretrain/cav_base_audioset_pretrain_${pretrain_path}/${pretrain_dataset}/model_checkpoint_0015.pth \
cl_algo.name=Finetune_head \
data.target_task=['All'] \
data.args.use_audio=True \
data.args.video_duration=4. \
data.args.audio_duration=10. \
data.args.num_frames=4 \
logging.eval_freq=1 \
logging.name=cav_base_audioset_finetune_head \
logging.suffix=_${pretrain_path}_${pretrain_dataset}_${trial} \
logging.save_freq_mints=120 \
logging.print_freq=20 \
logging.save_freq=15 \
logging.resume_path='' \
environment.workers=20 \
environment.slurm=False \
environment.world_size=1 \
environment.ngpu=4 \
environment.multiprocessing_distributed=True \
environment.distributed=True \
environment.dist_url=env:// \
environment.rank=-1 \
optim=adamw \
optim.layer_decay=0.75 \
optim.args.lr=5e-4 \
optim.args.betas=[0.95,0.999] \
optim.args.weight_decay=5e-6 \
optim.epochs=15 \
optim.batch_size=36 \
optim.per_gpu_batchsize=9 \
optim.use_lr_scheduler=True \
optim.lr_scheduler=True \
optim.lr_scheduler=WarmupCosineSchedule \
optim.lr_scheduler_args.warmup_epochs=2 \
optim.lr_scheduler_args.final_lr_ratio=0. \
