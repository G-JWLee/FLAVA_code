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
  echo "Default memory size is 5000"
  memory_size=5000
else
  echo "$4 is given as memory size"
  memory_size=$4
fi

if [ "$5" == "" ]; then
  echo "Default video compression ratio is 0.5"
  video_comp_ratio=0.5
else
  echo "$5 is given as video compression ratio"
  video_comp_ratio=$5
fi

if [ "$6" == "" ]; then
  echo "Default audio compression ratio is 0.5"
  audio_comp_ratio=0.5
else
  echo "$6 is given as audio compression ratio"
  audio_comp_ratio=$6
fi

if [ "$7" == "" ]; then
  echo "Default alpha is 0.5"
  alpha=0.5
else
  echo "$7 is given as alpha"
  alpha=$7
fi

if [ "$8" == "" ]; then
  echo "Default att_temperature is 0.1"
  att_temperature=0.1
else
  echo "$8 is given as att_temperature"
  att_temperature=$8
fi

export CUDA_VISIBLE_DEVICES=${gpus}
PYTHONPATH=. python main.py --config-name=cav_pretrain_audioset -m \
environment.port=21100 \
backbone=cav \
backbone.args.mask_ratio_a=0.8 \
backbone.args.mask_ratio_v=0.8 \
criterion=fa_mae_cont_vam \
criterion.args.norm_pix_loss=True \
criterion.args.load_local_path=${home_path}/FLAVA_code/baseline_ckpt/vggsound_pretrained.pth \
criterion.args.get_va_recall_metric=True \
cl_algo=ours_pp_der_pp_spurious_key_query_soft39 \
cl_algo.args.avm_pretrain_path=${home_path}/FLAVA_code/baseline_ckpt/vggsound_pretrained_submodule.pth \
cl_algo.args.core_video_ratio=${video_comp_ratio} \
cl_algo.args.core_audio_ratio=${audio_comp_ratio} \
cl_algo.args.num_core_audio_times=4 \
cl_algo.args.mem_args.memory_size=${memory_size} \
cl_algo.args.alpha=${alpha} \
cl_algo.args.att_temperature=${att_temperature} \
data_augm=cav_augm \
data.target_task=['human','vehicle','nature','animal','others','home','music'] \
data.skip_task=[] \
data.args.video_duration=4. \
data.args.audio_duration=10. \
data.args.use_audio=True \
data.args.num_frames=4 \
logging.eval_freq=3 \
logging.retrieve_freq=3 \
logging.resume_path='' \
logging.name=cav_base_audioset_pretrain \
logging.suffix=_flava_${random_seed}_${memory_size} \
logging.save_freq_mints=120 \
logging.print_freq=20 \
logging.save_freq=15 \
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
optim.epochs=15 \
optim.batch_size=36 \
optim.per_gpu_batchsize=9 \
optim.layer_decay=1.0 \
optim.use_lr_scheduler=False \
