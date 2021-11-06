run_glue () {
    VENV=env-$1

    if [ ! -d $VENV ]
   
    then
        virtualenv --system-site-packages -p python3 $VENV
        source $VENV/bin/activate

        echo $(which python)
        echo $(which pip)

        pip install -r requirements.txt
    fi

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
        --patches $PATCHES 2>&1
}

run_one_model () {
    export MODEL=$1/checkpoint-$2
    export OUTPUT=$1-$2

    run_glue RTE

    echo DONE
}

export BATCH=16
export PATCHES='none'

run_three () {
    run_one_model rec $1
    run_one_model mix $1
    run_one_model mlm $1
}

run_three 1
run_three 2
run_three 4
run_three 8
run_three 16
run_three 32
run_three 64
run_three 128
run_three 256
run_three 512
run_three 1024
run_three 2048
run_three 4096
run_three 8192
run_three 16384
run_three 32768
run_three 65536
run_three 131072
run_three 262144
