source env/bin/activate

TRAIN_FILE=$1

echo Trainig on $TRAIN_FILE

python run_language_modeling.py \
    --output_dir=bert \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --mlm \
    --do_train \
    --train_data_file=$TRAIN_FILE

python run_language_modeling.py \
    --output_dir=bert-blind \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --custom \
    --blind

python run_language_modeling.py \
    --output_dir=bert-ortho \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --custom \
    --ortho

python run_language_modeling.py \
    --output_dir=bert-blortho \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --custom \
    --blind --ortho
