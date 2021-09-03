source eval.sh

export MODEL=bert-blind-small-resume-checkpoint/checkpoint-20714/
export BATCH=16
export PATCHES="none"

export TASK=SST-2
export OUTPUT=resume-$TASK
run_glue

export TASK=STS-B
export OUTPUT=resume-$TASK
run_glue

export TASK=QQP
export OUTPUT=resume-$TASK
run_glue

export TASK=MNLI
export OUTPUT=resume-$TASK
run_glue

export TASK=RTE
export OUTPUT=resume-$TASK
run_glue
