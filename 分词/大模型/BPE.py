# @Time: 2024/9/16 11:43
# @Author: xy
from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
# get dataset
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

# get generator
batch_size = 1000
texts_generator = (dataset[i: i + batch_size]["text"] for i in range(0, len(dataset), batch_size))

# load an empty tokenizer for training
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))