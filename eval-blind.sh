run_glue () {
    python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-blind \
        --tokenizer_name=bert-base-uncased \
        --task_name $1 \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir glue/$1 \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir bert-blind-glue-original-$1 \
        --save_steps 100000000 \
        --model_version 1
}

run_glue CoLA
run_glue SST-2
run_glue STS-B
run_glue QQP
run_glue MNLI
run_glue QNLI
run_glue RTE
