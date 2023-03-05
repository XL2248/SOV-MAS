#!/bin/bash

# misc. settings
export seed=1234
#output=$1
# model settings
#export model_name=<path/to/trained/model/directory>
export model_name=/path/to/trained/model/directory/mix-shot_multilingual_multimodal_multigpu_cross_attn_type4_fusion_layer11_lr1_headnum8_imglen256_box_order_v0_multitask_mask_alpha1.0_beta1.0/



# batch / sequence sizes
export PER_DEVICE_EVAL_BATCH_SIZE=8
export MAX_SOURCE_LENGTH=512
export TEST_MAX_TARGET_LENGTH=84

# evaluation settings
#export rouge_lang="$2"
export eval_beams=4
export length_penalty=0.6
export no_repeat_ngram_size=2
export HOME=/path/to/code/directory/code4SOV-MAS
# optional_arguments
optional_arguments=(
    "--cache_dir cache_dir/"
)

langs="amharic chinese_traditional igbo marathi portuguese sinhala thai vietnamese arabic english indonesian nepali punjabi somali tigrinya welsh azerbaijani french oromo russian spanish turkish yoruba bengali gujarati kirundi pashto scottish_gaelic swahili ukrainian burmese hausa korean persian serbian_cyrillic tamil urdu chinese_simplified hindi kyrgyz pidgin serbian_latin telugu uzbek japanese"
export WANDB_DISABLED=true
code=/path/to/code/directory/code4SOV-MAS
cross_attn_type=4
fusion_layer=11
for lang in $langs
do
    export rouge_lang="$lang"
    export input_dir=/path/to/input_data_dir/individual_img/$lang
    export output_dir=/path/to/output_dir/individual_img/$lang
    echo "lang", $lang
    python $code/pipeline.py \
        --model_name_or_path $model_name \
        --data_dir $input_dir --output_dir $output_dir \
        --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
        --max_source_length $MAX_SOURCE_LENGTH --test_max_target_length $TEST_MAX_TARGET_LENGTH \
        --rouge_lang $rouge_lang --length_penalty $length_penalty --no_repeat_ngram_size $no_repeat_ngram_size \
        --eval_beams $eval_beams --seed $seed --overwrite_output_dir --predict_with_generate --do_predict \
        --max_img_len=256 \
        --n_attn_heads=8 \
        --img_lr_factor=5 \
        --use_forget_gate \
        --cross_attn_type=${cross_attn_type} \
        --fusion_layer=${fusion_layer} \
        --dim_common=256 \
        --use_img_trans \
        $(echo ${optional_arguments[@]})
done
