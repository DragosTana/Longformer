from torch.utils.data import Dataset
import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from itertools import chain
import multiprocessing

raw_data = load_from_disk("data/my_wikipedia")

def batch_iterator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]   

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_processes = multiprocessing.cpu_count()

print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

