
# Wikihop model from:
# "Longformer: The Long-Document Transformer", Beltagy et al, 2020: https://arxiv.org/abs/2004.05150


# Before training, download and prepare the data. The data preparation step takes a few minutes to tokenize and save the data.
# (1) Download data from http://qangaroo.cs.ucl.ac.uk/
# (2) unzip the file `unzip qangaroo_v1.1.zip`.  This creates a directory `qangaroo_v1.1`.
# (3) Prepare the data (tokenize, etc): `python scripts/wikihop.py --prepare-data --data-dir /path/to/qarangoo_v1.1/wikihop`

# To train base model run:
#python scripts/wikihop.py --save-dir /path/to/output --save-prefix longformer_base_4096_wikihop --data-dir /path/to/qangaroo_v1.1/wikihop --model-name longformer-base-4096 --num-workers 1 --num-epochs 15
#
# Note: this is work-in-progress update of existing code and may still have bugs.



import json
import os
import time
import random
from itertools import chain

import torch
from torch.utils.data import Dataset




def normalize_string(s):
    s = s.replace(' .', '.')
    s = s.replace(' ,', ',')
    s = s.replace(' !', '!')
    s = s.replace(' ?', '?')
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    s = s.replace(" 's", "'s")
    return ' '.join(s.strip().split())


def get_wikihop_roberta_tokenizer(tokenizer_name='roberta-base'):
    from transformers import RobertaTokenizer
    additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]'] # add special tokens for wikihop
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(additional_tokens)

    return tokenizer


def preprocess_wikihop(infile, tokenizer_name='roberta-large', sentence_tokenize=False):
    import nltk
    nltk.download('punkt')
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

        instance['query_tokens'] = query_tokens
        instance['supports_tokens'] = supports_tokens
        instance['candidate_tokens'] = candidate_tokens
        instance['answer_index'] = answer_index

    print("Finished tokenizing")
    return data


def preprocess_wikihop_train_dev(rootdir, tokenizer_name='roberta-large', sentence_tokenize=False):
    for split in ['dev', 'train']:
        infile = os.path.join(rootdir, split + '.json')
        if sentence_tokenize:
            outfile = os.path.join(rootdir, split + '.sentence.tokenized.json')
        else:
            outfile = os.path.join(rootdir, split + '.tokenized.json')
        print("Processing {} split".format(split))
        data = preprocess_wikihop(infile, tokenizer_name=tokenizer_name, sentence_tokenize=sentence_tokenize)
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


if __name__ == '__main__':

    wikihop_train = WikihopQADataset(filepath='./data/wikihop/train.tokenized.json', 
                                     shuffle_candidates=True, 
                                     tokenize=False, 
                                     tokenizer_name='distilbert/distilroberta-base',
                                     sentence_tokenize=False)
    
    print(wikihop_train[0])
    #decode 
    print(wikihop_train._tokenizer.decode(wikihop_train[0][0]))
    print(wikihop_train._tokenizer.decode(wikihop_train[0][1]))
    print(wikihop_train[0][2])
    print(wikihop_train[0][3])


    