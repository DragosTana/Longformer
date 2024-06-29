import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
import torch
import re
from transformers import RobertaModel
from torch.utils.data import Dataset
from data import Hyperpartisan
from transformers import AutoTokenizer

import json
import os
import time
import random
import numpy as np
from itertools import chain
import torch
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

from model.config import LongformerConfig
from model.longformer import Longformer 
from model.sliding_chunks import pad_to_window_size
from torch import optim
from trainer import Trainer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_string(s):
    """
    Normalize the string by removing extra spaces and fixing common punctuation issues.
    """
    s = s.replace(' .', '.')
    s = s.replace(' ,', ',')
    s = s.replace(' !', '!')
    s = s.replace(' ?', '?')
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    s = s.replace(" 's", "'s")
    return ' '.join(s.strip().split())

def get_wikihop_roberta_tokenizer(tokenizer_name='roberta-base'):
    """add [question], [/question], [ent], [/ent] special tokens to the tokenizer"""
    additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]']  # add special tokens for wikihop
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(additional_tokens)
    return tokenizer

def preprocess_wikihop(infile, tokenizer_name='roberta-large', sentence_tokenize=False, max_length=2048):
    from nltk.tokenize import sent_tokenize

    tokenizer = get_wikihop_roberta_tokenizer(tokenizer_name)

    def tok(s):
        return tokenizer.tokenize(normalize_string(s), add_prefix_space=True)

    def sent_tok(s):
        return tokenizer.tokenize(''.join(['<s> ' + sent + '</s>' for sent in sent_tokenize(normalize_string(s))]), add_prefix_space=False)

    if sentence_tokenize:
        the_tok = sent_tok
        doc_start = '<doc-s>'
        doc_end = '</doc-s>'
    else:
        the_tok = tok
        doc_start = '</s>'
        doc_end = '</s>'

    with open(infile, 'r') as fin:
        data = json.load(fin)
    print("Read data, {} instances".format(len(data)))

    t1 = time.time()
    filtered_data = []
    for instance_num, instance in enumerate(data):
        if instance_num % 100 == 0:
            print("Finished {} instances of {}, total time={}".format(instance_num, len(data), time.time() - t1))
        
        query_tokens = ['[question]'] + the_tok(instance['query']) + ['[/question]']
        supports_tokens = [
            [doc_start] + the_tok(support) + [doc_end]
            for support in instance['supports']
        ]
        candidate_tokens = [
            ['[ent]'] + the_tok(candidate) + ['[/ent]']
            for candidate in instance['candidates']
        ]
        answer_index = instance['candidates'].index(instance['answer'])

        combined_length = len(query_tokens) + sum(len(tokens) for tokens in supports_tokens) + sum(len(tokens) for tokens in candidate_tokens)
        
        if combined_length <= max_length:
            instance['query_tokens'] = query_tokens
            instance['supports_tokens'] = supports_tokens
            instance['candidate_tokens'] = candidate_tokens
            instance['answer_index'] = answer_index
            filtered_data.append(instance)

    print("Finished tokenizing. Filtered data contains {} instances".format(len(filtered_data)))
    return filtered_data

def preprocess_wikihop_train_dev(rootdir, tokenizer_name='roberta-large', sentence_tokenize=False, max_length=2048):
    for split in ['dev', 'train']:
        infile = os.path.join(rootdir, split + '.json')
        if sentence_tokenize:
            outfile = os.path.join(rootdir, split + '.sentence.tokenized.json')
        else:
            outfile = os.path.join(rootdir, split + '.tokenized_2048.json')
        print("Processing {} split".format(split))
        data = preprocess_wikihop(infile, tokenizer_name=tokenizer_name, sentence_tokenize=sentence_tokenize, max_length=max_length)
        with open(outfile, 'w') as fout:
            fout.write(json.dumps(data))



class WikihopQADataset(Dataset):
    def __init__(self, filepath, shuffle_candidates, tokenize=False, tokenizer_name='roberta-large', sentence_tokenize=False):
        super().__init__()

        if not tokenize:
            print("Reading cached data from {}".format(filepath))
            with open(filepath, 'r') as fin:
                self.instances = json.load(fin)
        else:
            print("Pre-processing data from {}".format(filepath))
            self.instances = preprocess_wikihop(filepath, tokenizer_name=tokenizer_name, sentence_tokenize=sentence_tokenize)

        self.shuffle_candidates = shuffle_candidates
        self._tokenizer = get_wikihop_roberta_tokenizer(tokenizer_name)

    @staticmethod
    def collate_single_item(x):
        # for batch size = 1
        assert len(x) == 1
        return [x[0][0].unsqueeze(0), x[0][1].unsqueeze(0), x[0][2], x[0][3]]
        return x
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self._convert_to_tensors(self.instances[idx])

    def _convert_to_tensors(self, instance):
        # list of wordpiece tokenized candidates surrounded by [ent] and [/ent]
        candidate_tokens = instance['candidate_tokens']
        # list of word piece tokenized support documents surrounded by </s> </s>
        supports_tokens = instance['supports_tokens']
        query_tokens = instance['query_tokens']
        answer_index = instance['answer_index']

        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))

        # concat all the candidate_tokens with <s>: <s> + candidates
        all_candidate_tokens = ['<s>'] + query_tokens

        # candidates
        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))
        if self.shuffle_candidates:
            random.shuffle(sort_order)
            new_answer_index = sort_order.index(answer_index)
            answer_index = new_answer_index
        all_candidate_tokens.extend(chain.from_iterable([candidate_tokens[k] for k in sort_order]))

        # the supports
        n_supports = len(supports_tokens)
        sort_order = list(range(n_supports))
        if self.shuffle_candidates:
            random.shuffle(sort_order)
        all_support_tokens = list(chain.from_iterable([supports_tokens[k] for k in sort_order]))

        # convert to ids
        candidate_ids = self._tokenizer.convert_tokens_to_ids(all_candidate_tokens)
        support_ids = self._tokenizer.convert_tokens_to_ids(all_support_tokens)

        # get the location of the predicted indices
        predicted_indices = [k for k, token in enumerate(all_candidate_tokens) if token == '[ent]']

        # candidate_ids, support_ids, prediction_indices, correct_prediction_index
        return torch.tensor(candidate_ids), torch.tensor(support_ids), torch.tensor(predicted_indices), torch.tensor([answer_index])
        #return {"candidate_ids": torch.tensor(candidate_ids), "support_ids": torch.tensor(support_ids), "predicted_indices": torch.tensor(predicted_indices), "correct_prediction_index": torch.tensor([answer_index])}

train_dataset = WikihopQADataset("data/wikihop/train.tokenized.json", shuffle_candidates=False)
lengths = []    
count = 0
for i in tqdm(range(len(train_dataset))):
    lengths.append(len(train_dataset[i][0]) + len(train_dataset[i][1]))
    if len(train_dataset[i][0]) + len(train_dataset[i][1]) > 512:
        count += 1
    
plt.hist(lengths, bins=100)
plt.title("Length of input tokens")
plt.xlabel("Number of tokens")
plt.ylabel("Number of instances")
print(f"Number of instances with more than 512 tokens: {count/len(train_dataset)*100:.2f}%")
plt.show()

#preprocess_wikihop_train_dev("data/wikihop", tokenizer_name='distilbert/distilroberta-base', sentence_tokenize=False)