export CKPT_TYPE=$1

run_glue () {
    VENV=env-$1

    if [ ! -d $VENV ]
    then
        virtualenv --system-site-packages -p python3 $VENV
        pip install -r requirements.txt
    fi

    source $VENV/bin/activate
    echo $(which python)
    echo $(which pip)

    python run_glue.py \
        --model_type bert-base-uncased \
        --model_name_or_path $MODEL \
        --tokenizer_name=bert-base-uncased \
        --task_name $1 \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir glue/$1 \
        --max_seq_length 128 \
        --per_gpu_train_batch_size $BATCH \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $OUTPUT-$1 \
        --save_steps 1000000000 \
        --eval_all_checkpoints \
        --fp16 \
        --patches $PATCHES 2>&1
    
    deactivate
}

run_one_model () {
    export MODEL=$1/checkpoint-$2
    export OUTPUT=$1-$2

    run_glue MNLI
    run_glue QNLI
    run_glue QQP
    run_glue RTE
    run_glue SST-2
    run_glue STS-B

    echo DONE
}

export BATCH=16
export PATCHES='none'

run_checkpoint () {
    run_one_model CKPT_TYPE $1
}

run_checkpoint 0
run_checkpoint 1
run_checkpoint 2
run_checkpoint 4
run_checkpoint 8
run_checkpoint 16
run_checkpoint 32
run_checkpoint 64
run_checkpoint 128
run_checkpoint 256
run_checkpoint 512
run_checkpoint 1024
run_checkpoint 2048
run_checkpoint 4096
run_checkpoint 8192
run_checkpoint 16384
run_checkpoint 32768
run_checkpoint 65536
run_checkpoint 131072
run_checkpoint 262144
