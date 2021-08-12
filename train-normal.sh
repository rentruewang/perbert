export TRAIN_FILE=$1
python run_language_modeling.py \
    --output_dir=bert \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps -1 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm 
