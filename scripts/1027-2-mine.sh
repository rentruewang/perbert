MODEL_PARENT=my-model
VENV=env-$MODEL_PARENT-2
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

run_all () {
	run_glue CoLA 
	run_glue MNLI
	run_glue QNLI
	run_glue QQP
	run_glue SST-2
	run_glue STS-B

    echo DONE
}

export BATCH=8
export PATCHES='none'

export MODEL=$MODEL_PARENT/checkpoint-58420
export OUTPUT=$MODEL_PARENT-1027-58420
run_all

export MODEL=$MODEL_PARENT/checkpoint-52578
export OUTPUT=$MODEL_PARENT-1027-52578
run_all

export MODEL=$MODEL_PARENT/checkpoint-46736
export OUTPUT=$MODEL_PARENT-1027-46736
run_all

export MODEL=$MODEL_PARENT/checkpoint-40894
export OUTPUT=$MODEL_PARENT-1027-40894
run_all

export MODEL=$MODEL_PARENT/checkpoint-35052
export OUTPUT=$MODEL_PARENT-1027-35052
run_all
