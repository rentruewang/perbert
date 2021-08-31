import subprocess

template = """
source eval.sh

export TASK={task}
export CHECKPOINT={checkpoint}
export MODEL=bert-{model}/$CHECKPOINT
export BATCH={batch}
export PATCHES="none"
export OUTPUT=bert-glue-{model}-$TASK-$CHECKPOINT
run_glue | bat -l=log --paging=never --style=plain
"""

command = template.split("\n")
command = ";".join(line for line in command if line)


# for task in "SST-2 STS-B QQP MNLI RTE".split():
for task in ["SST-2"]:
    for model in ["original-10", "phased-1"]:
        for checkpoint in (
            "checkpoint-24702",
            "checkpoint-123510",
            "checkpoint-222318",
            "checkpoint-49404",
            "checkpoint-148212",
            "checkpoint-247020",
            "checkpoint-74106",
            "checkpoint-172914",
            "checkpoint-98808",
            "checkpoint-197616",
        ):
            args = {"task": task, "checkpoint": checkpoint, "model": model, "batch": 8}
            subprocess.call(command.format(**args), shell=True)
