source env/bin/activate

TRAIN_FILE=$1

echo Trainig on $TRAIN_FILE, running bert

python run_language_modeling.py \
    --output_dir=bert \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --mlm \
    --do_train \
    --max_steps 1 \
    --train_data_file=$TRAIN_FILE

echo done
echo
# echo Trainig on $TRAIN_FILE, running bert-blind

# python run_language_modeling.py \
#     --output_dir=bert-blind \
#     --model_type=bert-base-uncased \
#     --model_name_or_path=bert-base-uncased \
#     --do_train \
#     --train_data_file=$TRAIN_FILE \
#     --custom \
#     --blind 

# echo done
# echo
# echo Trainig on $TRAIN_FILE, running bert-ortho

# python run_language_modeling.py \
#     --output_dir=bert-ortho \
#     --model_type=bert-base-uncased \
#     --model_name_or_path=bert-base-uncased \
#     --do_train \
#     --train_data_file=$TRAIN_FILE \
#     --custom \
#     --ortho 

# echo done
# echo
# echo Trainig on $TRAIN_FILE, running bert-blortho

# python run_language_modeling.py \
#     --output_dir=bert-blortho \
#     --model_type=bert-base-uncased \
#     --model_name_or_path=bert-base-uncased \
#     --do_train \
#     --train_data_file=$TRAIN_FILE \
#     --max_steps 1 \
#     --custom \
#     --blind --ortho 

# echo done
# echo
