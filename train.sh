source env/bin/activate

TRAIN_FILE=~/Dropbox/Tmp/bert-pretraining.txt
python run_language_modeling.py \
    --output_dir=output \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE
