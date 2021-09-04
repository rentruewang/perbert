python run_language_modeling.py \
    --output_dir=emb-gel-1 \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 24 \
    --do_train \
    --train_data_file=bert-pretraining.txt \
    --patches embedding gelu-1 save-10 fast-terminate

python run_language_modeling.py \
    --output_dir=emb-gel-2 \
    --model_type=bert-base-uncased \
    --model_name_or_path=bert-base-uncased \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 24 \
    --do_train \
    --train_data_file=bert-pretraining.txt \
    --patches embedding gelu-2 save-10 fast-terminate

python run_language_modeling.py \
    --output_dir=bert-by-gel-1 \
    --model_type=bert-base-uncased \
    --model_name_or_path=emb-gel-1 \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 24 \
    --do_train \
    --train_data_file=bert-pretraining.txt \
    --patches none save-10 fast-terminate

python run_language_modeling.py \
    --output_dir=bert-by-gel-2 \
    --model_type=bert-base-uncased \
    --model_name_or_path=emb-gel-2 \
    --tokenizer_name=bert-base-uncased \
    --save_steps 100000000 \
    --per_gpu_train_batch_size 24 \
    --do_train \
    --train_data_file=bert-pretraining.txt \
    --patches none save-10 fast-terminate
