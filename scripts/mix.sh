export MODEL=mix
export OUTPUT=$MODEL

export ENV=env-$OUTPUT
virtualenv $ENV -p python3

echo "Sourcing env-$MODEL"
source $ENV/bin/activate

echo $(which python)
echo $(which pip)

pip install -r requirements.txt

python run_language_modeling.py \
      --output_dir=$MODEL \
      --model_type=bert-base-uncased \
      --model_name_or_path=bert-base-uncased \
      --tokenizer_name=bert-base-uncased \
      --save_steps 100000000 \
      --per_gpu_train_batch_size 24 \
      --do_train \
      --train_data_file=bert-pretraining.txt \
      --patches save10 savelog maskcross anddecay mlmpair firstlayer

rm -r $ENV
# run_glue () {
#     python run_glue.py \
#         --model_type bert-base-uncased \
#         --model_name_or_path $MODEL \
#         --tokenizer_name=bert-base-uncased \
#         --task_name $1 \
#         --do_train \
#         --do_eval \
#         --do_lower_case \
#         --data_dir glue/$1 \
#         --max_seq_length 128 \
#         --per_gpu_train_batch_size $BATCH \
#         --learning_rate 2e-5 \
#         --num_train_epochs 3.0 \
#         --output_dir $OUTPUT-$1 \
#         --save_steps 1000000000 \
#         --patches $PATCHES 2>&1
# }

# run_all () {
# 	run_glue CoLA 
# 	run_glue MNLI
# 	run_glue QNLI
# 	run_glue QQP
# 	run_glue SST-2
# 	run_glue STS-B

#     echo DONE
# }

# export BATCH=8
# export PATCHES='none'

# run_all
