VENV=env
virtualenv -p python3 $VENV
source $VENV/bin/activate

echo $(which python)
echo $(which pip)

pip install -r requirements.txt

# python run_language_modeling.py \
#       --output_dir=$MODEL \
#       --model_type=bert-base-uncased \
#       --model_name_or_path=bert-base-uncased \
#       --tokenizer_name=bert-base-uncased \
#       --save_steps 100000000 \
#       --per_gpu_train_batch_size 24 \
#       --do_train \
#       --train_data_file=bert-pretraining.txt \
#       --patches save10 smallsubset earlyfocus maskcross anddecay mlmpair firstlayer

run_glue () {
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
        run_glue CoLA
        run_glue MNLI
        run_glue QNLI
        run_glue QQP
        run_glue SST-2
        run_glue STS-B

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
