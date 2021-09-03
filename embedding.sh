export TRAIN_FILE=bert-pretraining.txt

python run_language_modeling.py \
    --output_dir=embedding \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 64 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --patches embedding small-subset fast-terminate

python run_language_modeling.py \
    --output_dir=embedding-gelu-1 \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 64 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --patches embedding small-subset fast-terminate gelu-1

python run_language_modeling.py \
    --output_dir=embedding-gelu-2 \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 64 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --patches embedding small-subset fast-terminate gelu-2
