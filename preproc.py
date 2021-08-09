# from transformers.tokenization_bert import BertTokenizerFast
# from run_language_modeling import *
import os
import pickle

from tqdm import tqdm, trange
from transformers import BertTokenizerFast

file_path = "./bert-pretraining.txt"
model_type = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_type)
(directory, filename) = os.path.split(file_path)
print("Creating features from dataset file at {}".format(directory))
block_size = 512
lines_size = 1000000

with open(file_path, encoding="utf-8") as f:
    text = f.readlines()

directory = os.path.join(directory, "dataset-" + filename.split(".")[0])
os.makedirs(directory, exist_ok=True)

for (i, idx) in enumerate(range(0, len(text), lines_size)):
    fp = f"{i:04d}" + "." + filename.split(".")[1]
    cached_features_file = os.path.join(
        directory,
        model_type + "_cached_lm_" + str(block_size) + "_" + fp,
    )
    if os.path.exists(cached_features_file):
        print(cached_features_file + " exists. Skip.")
        continue
    print("loading " + fp)
    portion = text[idx : idx + lines_size]
    print("writing files, length =", len(portion))
    with open(os.path.join(directory, fp), "w+") as f:
        f.writelines(tqdm(portion))

    print("Creating features from dataset file at {}".format(directory))

    examples = []
    print("joining text")
    txt = "\n".join(tqdm(portion))

    print("tokenizing")
    tokens = tokenizer.tokenize(txt)
    print("converting")
    tokenized_text = tokenizer.convert_tokens_to_ids(tqdm(tokens))

    print("building")
    for i in trange(
        0, len(tokenized_text) - block_size + 1, block_size
    ):  # Truncate in block of block_size
        examples.append(
            tokenizer.build_inputs_with_special_tokens(
                tokenized_text[i : i + block_size]
            )
        )
    # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
    # If your dataset is small, first you should loook for a bigger one :-) and second you
    # can change this behavior by adding (model specific) padding.

    print("Saving features into cached file %s" % cached_features_file)
    with open(cached_features_file, "wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
# tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

# for i in range(
#     0, len(tokenized_text) - block_size + 1, block_size
# ):  # Truncate in block of block_size
#     examples.append(
#         tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
#     )
#     # print(examples[-1])
# Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
# If your dataset is small, first you should loook for a bigger one :-) and second you
# can change this behavior by adding (model specific) padding.

# print("Saving features into cached file %s", cached_features_file)
# with open(cached_features_file, "wb") as handle:
#     pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
