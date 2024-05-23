export PATH=your_env_path
export HF_DATASETS_CACHE=your_cache_dir
export NCCL_DEBUG=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=5
export NCCL_IB_DISABLE=1

run_version=${1:-'demo'}

#wandb_host=${2:-"https://api.wandb.ai"}  # Here is wandb official host, it can also be your local deployed wandb-service.
#wandb_key=${3:-""}
#wandb_project_name=${4:-""}

model_name_or_path=${5:-'qihoo360/360Zhinao-search'}
train_data=${6:-'data/toy_finetune_data.json'}
logging_steps=${7:-'1000'}
save_steps=${8:-'1000'}
per_device_train_batch_size=${9:-'8'}
gradient_accumulation_steps=${10:-'1'}
learning_rate=${11:-'1e-5'}
num_train_epochs=${12:-'5'}
train_group_size=${13:-'9'}

query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章："
output_dir="outputs/$run_version"

master_port=$(($$ % 1000 + 1025))
echo "master_port: ${master_port}"

# mkdir for logdir
logdir="log"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir"
fi

python -m torch.distributed.launch --nproc_per_node=8 --master_port=${master_port} \
      run.py \
      --deepspeed ds_config_zero2.json \
      --fp16 \
      --output_dir=${output_dir} \
      --wandb_host=${wandb_host} \
      --wandb_key=${wandb_key} \
      --wandb_project_name=${wandb_project_name} \
      --model_name_or_path=${model_name_or_path} \
      --train_data=${train_data} \
      --logging_steps=${logging_steps} \
      --save_steps=${save_steps} \
      --do_train \
      --per_device_train_batch_size=${per_device_train_batch_size} \
      --gradient_accumulation_steps=${gradient_accumulation_steps} \
      --learning_rate=${learning_rate} \
      --num_train_epochs=${num_train_epochs} \
      --query_instruction_for_retrieval=${query_instruction_for_retrieval} \
      --train_group_size=${train_group_size} \
      > ${logdir}/${run_version}.log 2>&1 &

