#!/bin/bash

#source activate "${BASE_DIR}/env"
# for ozstar only; the model must
# be cached if this variable is set
export LINK_CACHE_ONLY=true 
export cross_attn_type=$1
export fusion_layer=$2
export data_type=$3
export lr=$4
export head_num=$5
export img_len=$6
# training settings
export max_steps=35000
export save_steps=2500
export logging_steps=100

# validation settings
export evaluation_strategy="no"

# model settings
#export model_name="google/mt5-base"
export model_name="/path/to/trained/model/checkpoint/mmt5-base" # downloaded mt5-base and rename it to mmt5-base for launch the "multimodal mt5" model.
if false; then
if [[ "${SLURM_PROCID:-0}" -eq 0 && "${SLURM_LOCALID:-0}" -eq 0 ]]; then
    mkdir -p $OUTPUT_DIR
    python "${BASE_DIR}/generate_data.py" \
        --dataset_dir $ROOT_DATASET_DIR \
        --output_dir $INPUT_DIR \
        --training_type $TRAINING_TYPE \
        --pivot_lang $PIVOT_LANG \
        --exclude_native $EXCLUDE_NATIVE \
        --min_example_count $MIN_EXAMPLE_COUNT
fi
fi
# optimization settings
export learning_rate=${lr}
export warmup_steps=5000
export gradient_accumulation_steps=8
export weight_decay=0.01
export lr_scheduler_type="transformer"
export label_smoothing_factor=0.1
export upsampling_factor=0.5
# misc. settings
export seed=1234
export BASE_DIR="/path/to/input_data_dir"
# input / output settings
export input_dir="${BASE_DIR}/XLSum_input/${data_type}"
export output_dir="${BASE_DIR}/XLSum_output/${data_type}_multilingual_multimodal_multigpu_cross_attn_type${cross_attn_type}_fusion_layer${fusion_layer}_lr${lr}_headnum${head_num}_imglen${img_len}_box_order_v0_multitask_mask_upsamp${upsampling_factor}"
if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi
# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export MAX_SOURCE_LENGTH=512
export MAX_TARGET_LENGTH=84
# optional arguments
optional_arguments=(
    "--logging_first_step"
    "--cache_dir ${BASE_DIR}/cache_dir"
)

export WANDB_WATCH=false
export WANDB_MODE="dryrun"
export WANDB_DISABLED=true
export HDF5_USE_FILE_LOCKING=false
export NGPUS=8
export code=/path/to/code/directory/code4SOV-MAS
NPROC_PER_NODE=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                --nproc_per_node=${NPROC_PER_NODE} \
                "${code}/pipeline.py" \
    --model_name_or_path $model_name \
    --data_dir $input_dir --output_dir $output_dir \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
    --upsampling_factor $upsampling_factor --seed $seed \
    --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy --do_train \
    --max_img_len=${img_len} \
    --n_attn_heads=${head_num} \
    --img_lr_factor=${lr} \
    --use_forget_gate \
    --cross_attn_type=${cross_attn_type} \
    --fusion_layer=${fusion_layer} \
    --dim_common=256 \
    --use_img_trans \
    --ignore_data_skip \
    --alphs 1.0 \
    --beta 1.0 \
    $(echo ${optional_arguments[@]})

