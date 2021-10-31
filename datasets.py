import os

from tqdm import tqdm

with open("bert-pretraining.txt") as f:
    text = f.readlines()

MAX_FILE_SIZE = 1_000_000

file_contents = []
for idx in tqdm(range(0, len(text), MAX_FILE_SIZE)):
    file_contents.append(text[idx : idx + MAX_FILE_SIZE])

DIR = "dataset-bert-pretraining"
os.makedirs(DIR, exist_ok=True)

for (idx, data) in enumerate(tqdm(file_contents)):
    with open(f"{DIR}/{idx:04d}.txt", "w") as f:
        f.writelines(map(lambda s: s + "\n", data))
