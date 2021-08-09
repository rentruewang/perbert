source env/bin/activate
pip install -r requirements.txt
export TRAIN_FILE=$1
python run_language_modeling.py \
    --output_dir=bert \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --mlm \
    --save_steps -1 \
    --do_train \
    --train_data_file=$TRAIN_FILE

exit
export GLUE_DIR=./glue

export TASK_NAME=CoLA

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/


export TASK_NAME=SST-2

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/



export TASK_NAME=STS-B

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/



export TASK_NAME=QQP

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/


export TASK_NAME=MNLI

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/


export TASK_NAME=QNLI

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/


export TASK_NAME=RTE

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/


export TASK_NAME=WNLI

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name=bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert-glue/$TASK_NAME/
