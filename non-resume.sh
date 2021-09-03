source eval.sh

export MODEL=bert-orig-small-mlm/checkpoint-20714/
export BATCH=16
export PATCHES="none"

export TASK=SST-2
export OUTPUT=no-resume-orig-mlm-$TASK
run_glue

export TASK=STS-B
export OUTPUT=no-resume-orig-mlm-$TASK
run_glue

export TASK=QQP
export OUTPUT=no-resume-orig-mlm-$TASK
run_glue

export TASK=MNLI
export OUTPUT=no-resume-orig-mlm-$TASK
run_glue

export TASK=RTE
export OUTPUT=no-resume-orig-mlm-$TASK
run_glue
