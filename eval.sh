run_glue () {
    python run_glue.py \
        --model_type bert \
        --model_name_or_path $MODEL \
        --tokenizer_name=bert-base-uncased \
        --task_name $TASK \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir glue/$TASK \
        --max_seq_length 128 \
        --per_gpu_train_batch_size $BATCH \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $OUTPUT \
        --save_steps 1000000000 \
        --patches $PATCHES 2>&1
}
