export TRAIN_FILE=bert-pretraining.txt
python run_language_modeling.py \
    --output_dir=bert-blind-smaller-mlm \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 16 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --patches blindspot small-subset save-10
