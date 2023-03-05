#!/bin/bash
# to use this script for multilingual setup, use the variables 
# from `distributed_trainer.sh`
data=/path/to/input_data_dir/
export HOME=/path/to/code/directory
export CUDA_VISIBLE_DEVICES=0 
export code=/path/to/code/directory/code4SOV-MAS
#cross_attn_type=$1
lang=$1
alpha=1.0
# training settings
export num_train_epochs=20
export max_steps=0 # overrides epochs (must be 0 if using epochs)
export save_steps=500
export logging_steps=500

# validation settings
export evaluation_strategy="epoch" 
export evaluation_strategy="no"

# model settings
export ckpt=/path/to/trained/model/checkpoint/mt5-base/ # downloaded mt5-base model and rename it to mmt5-base for launch the "multimodal mt5" model.
# optimization settings
export learning_rate=5e-4
export warmup_steps=250 # we used 10% of the total number of steps as warmup for monolingual training.
export gradient_accumulation_steps=4
export weight_decay=0.01
export lr_scheduler_type="linear"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# input / output settings
export input_dir=${data}/XLSum_input/individual_img/${lang}
#export output_dir="XLSum_output/individual/bengali"
export output_dir=/path/to/output_dir/box_order_v0_multitask_lr5e4_ws250_${lang}_cl_mask_alpha${alpha}
if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export MAX_SOURCE_LENGTH=512
export MAX_TARGET_LENGTH=84
export VAL_MAX_TARGET_LENGTH=$MAX_TARGET_LENGTH
export PER_DEVICE_EVAL_BATCH_SIZE=8
export TEST_MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH}

# evaluation settings
#export rouge_lang="$2"
export eval_beams=4
export length_penalty=0.6
export no_repeat_ngram_size=2
# optional arguments
optional_arguments=(
    "--logging_first_step"
    "--cache_dir cache_dir/"
)

# optional for logging
export WANDB_DISABLED=true
export WANDB_WATCH=false
export WANDB_DISABLED=true
export HDF5_USE_FILE_LOCKING=false
rouge_lang=$lang
cross_attn_type=4
fusion_layer=11
python ${code}/pipeline.py \
    --model_name_or_path "$ckpt/checkpoint/mmt5-base" \
    --data_dir $input_dir --output_dir $output_dir \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH --logging_steps $logging_steps \
    --val_max_target_length $VAL_MAX_TARGET_LENGTH --seed $seed \
    --num_train_epochs=$num_train_epochs --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy --predict_with_generate --do_train --do_eval \
    --do_predict --test_max_target_length $TEST_MAX_TARGET_LENGTH --rouge_lang $rouge_lang --length_penalty $length_penalty --no_repeat_ngram_size $no_repeat_ngram_size --eval_beams $eval_beams \
    --max_img_len=256 \
    --n_attn_heads=8 \
    --img_lr_factor=5 \
    --use_forget_gate \
    --cross_attn_type=${cross_attn_type} \
    --fusion_layer=${fusion_layer} \
    --dim_common=256 \
    --use_img_trans \
    --ignore_data_skip \
    --alpha=${alpha} \
    --beta=${alpha} \
    $(echo ${optional_arguments[@]})
