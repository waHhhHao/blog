# @Time: 2024/9/22 16:17
# @Author: xy

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

sentences = ["embedding", "I love machine learning and nlp"]