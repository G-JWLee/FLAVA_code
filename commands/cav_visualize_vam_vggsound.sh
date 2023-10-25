#!/bin/bash

if [ "$1" == "vessl" ]; then
  export home_path="/input/jwlee"
  export home_data_path="/input/jwlee/dataset"
elif [ "$1" == "gcp" ]; then
  export home_path="/home/jwlee8877"
  export home_data_path="/home/jwlee8877/dataset"
elif [ "$1" == "nsml" ]; then
  export home_path="/mnt/hdd-nfs/jwlee"
  export home_data_path="/mnt/hdd-nfs/jwlee/dataset"
elif [ "$1" == "labv17" ]; then
  export home_path="/v17/jaewoo"
  export home_data_path="/d1/dataset"
elif [ "$1" == "labv6" ]; then
  export home_path="/c1/jwlee"
  export home_data_path="/v6/jaewoo/dataset"
else
  export home_path="/c1/jwlee"
  export home_data_path="/d1/dataset"
fi

base_model="$2"
dataset="$3"

if [ "$4" == "" ]; then
  echo "pretrained dataset sports"
  pretrain_dataset="sports"
else
  echo "pretrained dataset is $4"
  pretrain_dataset="$4"
fi

if [ "$5" == "" ]; then
  echo "CUDA_VISIBLE_DEVICES is 0 by default"
  gpus="0"
else
  echo "CUDA_VISIBLE_DEVICES is $5"
  gpus=$5
fi

if [ "$6" == "" ]; then
  echo "Default port is 21000"
  port=21000
else
  echo "$6 is given as port"
  port=$6
fi

export CUDA_VISIBLE_DEVICES=${gpus}

PYTHONPATH=. python main.py --config-name=cav_pretrain_vggsound -m \
environment.port=${port} \
train_algo=visualize \
train_algo.visualizer='attention_submodule_visualize' \
criterion=va_embedding \
criterion.args.load_local_path=${home_path}/TVLT_pytorch/experiments/checkpoints/cav_base_vggsound_pretrain/cav_base_vggsound_pretrain_${base_model}/${pretrain_dataset}/model_checkpoint_0010.pth \
cl_algo=ours_visualize_cl_spurious \
cl_algo.name=Ours_Visualize \
cl_algo.args.avm_pretrain_path=${home_path}/TVLT_pytorch/experiments/checkpoints/cav_base_vggsound_pretrain/cav_base_vggsound_pretrain_${base_model}/${pretrain_dataset}/model_checkpoint_0010.pth \
data_augm=cav_augm \
data_augm.video_data.name=Attention_vis_transform \
data_augm.audio_data.args.noise=False \
data.target_task=[${dataset}] \
data.args.debug=True \
data.args.sample_type=middle \
data.args.use_audio=True \
data.args.video_duration=4. \
data.args.audio_duration=10. \
logging.name=cav_base_vggsound_va_vis_${base_model} \
logging.suffix=_${dataset}_${pretrain_dataset} \
environment.seed=2021 \
environment.workers=20 \
environment.slurm=False \
environment.world_size=1 \
environment.ngpu=1 \
environment.multiprocessing_distributed=True \
environment.distributed=True \
environment.dist_url=env:// \
environment.rank=-1 \
optim.layer_decay=0 \
optim.per_gpu_batchsize=6 \
