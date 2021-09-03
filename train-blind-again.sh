python run_language_modeling.py \
    --output_dir=bert-blind-again \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 24 \
    --do_train \
    --train_data_file=bert-pretraining.txt \
    --patches small-subset blindspot save-10 fast-terminate
