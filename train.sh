export MODEL=$1
export OUTPUT=$MODEL
export PATCHES=${@:2}

echo "Model to save: $MODEL"
echo "Patches used: $PATCHES"

export VENV=env-$OUTPUT
virtualenv --system-site-packages $VENV -p python3

echo "Sourcing env-$MODEL"
source env-$OUTPUT/bin/activate

echo $(which python)
echo $(which pip)

pip install -r requirements.txt

python \
      run_language_modeling.py \
      --output_dir=$MODEL \
      --model_type=bert-base-uncased \
      --model_name_or_path=prajjwal1/bert-mini \
      --tokenizer_name=bert-base-uncased \
      --save_steps 100000000 \
      --per_gpu_train_batch_size 64 \
      --do_train \
      --train_data_file=bert-pretraining.txt \
      --patches $PATCHES

tar cvf $MODEL.tar.gz $MODEL

rm -r $VENV
