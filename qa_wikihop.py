
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
import numpy as np
from itertools import chain
import torch
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaModel, AutoModel, AutoTokenizer, RobertaConfig

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
    from transformers import RobertaTokenizer
    additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]'] # add special tokens for wikihop
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(additional_tokens)
    return tokenizer


def preprocess_wikihop(infile, tokenizer_name='roberta-large', sentence_tokenize=False):
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

def load_longformer(model_name='longformer-base-4096'):
    config = LongformerConfig.from_pretrained(model_name + '/')
    config.attention_mode = 'sliding_chunks'
    model = Longformer.from_pretrained(model_name + '/', config=config)

    # add four additional word embeddings for the special tokens
    current_embed = model.embeddings.word_embeddings.weight
    current_vocab_size, embed_size = current_embed.size()
    new_embed = model.embeddings.word_embeddings.weight.new_empty(current_vocab_size + 4, embed_size)
    new_embed.normal_(mean=torch.mean(current_embed).item(), std=torch.std(current_embed).item())
    new_embed[:current_vocab_size] = current_embed
    model.embeddings.word_embeddings.num_embeddings = current_vocab_size + 4
    del model.embeddings.word_embeddings.weight
    model.embeddings.word_embeddings.weight = torch.nn.Parameter(new_embed)

    return model

def load_distilroberta(model_name='distilbert/distilroberta-base'):
    
  
    model = AutoModel.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
    distil_roberta = RobertaModel(config, add_pooling_layer=False)
    distil_roberta.load_state_dict(model.state_dict(), strict=False)
    del model
    model = distil_roberta

    # add four additional word embeddings for the special tokens
    current_embed = model.embeddings.word_embeddings.weight
    current_vocab_size, embed_size = current_embed.size()
    new_embed = model.embeddings.word_embeddings.weight.new_empty(current_vocab_size + 4, embed_size)
    new_embed.normal_(mean=torch.mean(current_embed).item(), std=torch.std(current_embed).item())
    new_embed[:current_vocab_size] = current_embed
    model.embeddings.word_embeddings.num_embeddings = current_vocab_size + 4
    del model.embeddings.word_embeddings.weight
    model.embeddings.word_embeddings.weight = torch.nn.Parameter(new_embed)

    print("Loaded model")
    print(model)
    return model


def get_activations_longformer(model, candidate_ids, support_ids, max_seq_len, truncate_seq_len):
    # max_seq_len: the maximum sequence length possible for the model
    # truncate_seq_len: only use the first truncate_seq_len total tokens in the candidate + supports (e.g. just the first 4096)
    candidate_len = candidate_ids.shape[1]
    support_len = support_ids.shape[1]

    # attention_mask = 1 for local, 2 for global, 0 for padding (which we can ignore as always batch size=1)
    if candidate_len + support_len <= max_seq_len:
        token_ids = torch.cat([candidate_ids, support_ids], dim=1)
        attention_mask = torch.ones(token_ids.shape, dtype=torch.long, device=token_ids.device)
        # global attention to all candidates
        attention_mask[0, :candidate_len] = 2
        token_ids, attention_mask = pad_to_window_size(
            token_ids, attention_mask, model.config.attention_window[0], model.config.pad_token_id)

        return [model(token_ids, attention_mask=attention_mask)[0]]

    else:
        all_activations = []
        available_support_len = max_seq_len - candidate_len
        for start in range(0, support_len, available_support_len):
            end = min(start + available_support_len, support_len, truncate_seq_len)
            token_ids = torch.cat([candidate_ids, support_ids[:, start:end]], dim=1)
            attention_mask = torch.ones(token_ids.shape, dtype=torch.long, device=token_ids.device)
            # global attention to all candidates
            attention_mask[0, :candidate_len] = 2
            token_ids, attention_mask = pad_to_window_size(
                token_ids, attention_mask, model.config.attention_window[0], model.config.pad_token_id)

            activations = model(token_ids, attention_mask=attention_mask)[0]
            all_activations.append(activations)
            if end == truncate_seq_len:
                break

        return all_activations
    
def pad_to_max_length(input_ids, attention_mask, max_length, pad_token_id):
    """
    Pad input_ids and attention_mask to max_length
    """
    padding_length = max_length - input_ids.size(1)
    assert padding_length == max_length - attention_mask.size(1)
    if padding_length > 0:
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)
    return input_ids, attention_mask
    
def get_activations_roberta(model, candidate_ids, support_ids, max_seq_len, truncate_seq_len):
    """
    Handle the case where the input is too long for the model.
    """
    candidate_len = candidate_ids.shape[1]
    support_len = support_ids.shape[1]
    if candidate_len + support_len <= max_seq_len:
        token_ids = torch.cat([candidate_ids, support_ids], dim=1)
        attention_mask = torch.ones(token_ids.shape, dtype=torch.long, device=token_ids.device)
        token_ids, attention_mask = pad_to_max_length(
            token_ids, attention_mask, max_seq_len, model.config.pad_token_id)
        return [model(token_ids, attention_mask=attention_mask)[0]]
    else:
        all_activations = []
        available_support_len = max_seq_len - candidate_len
        for start in range(0, support_len, available_support_len):
            end = min(start + available_support_len, support_len, truncate_seq_len)
            token_ids = torch.cat([candidate_ids, support_ids[:, start:end]], dim=1)
            attention_mask = torch.ones(token_ids.shape, dtype=torch.long, device=token_ids.device)
            token_ids, attention_mask = pad_to_max_length(
                token_ids, attention_mask, max_seq_len, model.config.pad_token_id)
            activations = model(token_ids, attention_mask=attention_mask)[0]
            all_activations.append(activations)
            if end == truncate_seq_len:
                break
        return all_activations

class WikihopQAModel(nn.Module):
    def __init__(self):
        super(WikihopQAModel, self).__init__()

        self.distil_roberta = load_distilroberta("distilbert/distilroberta-base")
        self.answer_score = torch.nn.Linear(self.distil_roberta.embeddings.word_embeddings.weight.shape[1], 1, bias=False)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self._truncate_seq_len = 100000000000
            
    def forward(self, data, return_predicted_index=False):
        """
        We always consider batch size of one instance which has:
            question, candidate answers, supporting documents, and the correct answer.
        
        Input:
            candidate_ids: (1, candidate_length): <s> [question] question token ids [/question] [ent] candidate 1 token ids [/ent] [ent] candidate 2 ids ... [/ent]
            support_ids: (1, support_length): </s> document 1 token ids </s> </s> document 2 ids </s> ... </s> document M ids </s>
            predicted_indices = (1, num_candidates): [15, 22, 30, ..], a list of the indices in candidate_ids corresponding to the [ent] tokens used to predict logit for this candidate
            answer_index = (1, ) with the index of the correct answer
        """
        candidate_ids, support_ids, prediction_indices, correct_prediction_index = data
        activations = get_activations_roberta(
            self.distil_roberta,
            candidate_ids,
            support_ids,
            512,
            self._truncate_seq_len)
        
        prediction_activations = [act.index_select(1, prediction_indices) for act in activations]
        prediction_scores = [
            self.answer_score(prediction_act).squeeze(-1)
            for prediction_act in prediction_activations
        ]
        sum_prediction_scores = torch.cat(
                [pred_scores.unsqueeze(-1) for pred_scores in prediction_scores], dim=-1
        ).sum(dim=-1)
        loss = self.loss(sum_prediction_scores, correct_prediction_index)
        batch_size = candidate_ids.new_ones(1) * prediction_activations[0].shape[0]
        predicted_answers = sum_prediction_scores.argmax(dim=1)
        num_correct = (predicted_answers == correct_prediction_index).int().sum()
        
        if not return_predicted_index:
            return loss, batch_size, num_correct
        else:
            return loss, batch_size, num_correct, predicted_answers
        
    def train_step(self, batch, return_predicted_index=False):
        output = self.forward(batch, return_predicted_index)
        if return_predicted_index:
            loss, batch_size, num_correct, predicted_answers = output
            return loss, batch_size, num_correct, predicted_answers
        else:
            loss, batch_size, num_correct = output
            return loss, batch_size, num_correct
        
    def test_step(self, batch, return_predicted_index=False):
        output = self.forward(batch, return_predicted_index)
        if return_predicted_index:
            loss, batch_size, num_correct, predicted_answers = output
            return loss, batch_size, num_correct, predicted_answers
        else:
            loss, batch_size, num_correct = output
            return loss, batch_size, num_correct
        
        
if __name__ == "__main__":
    
    train_dataset = WikihopQADataset("data/wikihop/train.tokenized.json", shuffle_candidates=False)
    val_dataset = WikihopQADataset("data/wikihop/dev.tokenized.json", shuffle_candidates=False)
    
    # remove instances that are too long
    for instance in train_dataset.instances:
        if len(instance['query_tokens']) + sum(len(s) for s in instance['supports_tokens']) > 3500:
            train_dataset.instances.remove(instance)
    for instance in val_dataset.instances:
        if len(instance['query_tokens']) + sum(len(s) for s in instance['supports_tokens']) > 3500:
            val_dataset.instances.remove(instance)
    
    #select only 1000 instances for training
    #train_dataset.instances = train_dataset.instances[:500]
    #val_dataset.instances = val_dataset.instances[:500] 
    
    # batch size = 1 but we accumulate gradients
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=WikihopQADataset.collate_single_item)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=WikihopQADataset.collate_single_item)
    
    for instance in train_dataset.instances:
        if len(instance['query_tokens']) + sum(len(s) for s in instance['supports_tokens']) > 3500:
            train_dataset.instances.remove(instance)
    for instance in val_dataset.instances:
        if len(instance['query_tokens']) + sum(len(s) for s in instance['supports_tokens']) > 3500:
            val_dataset.instances.remove(instance)
    
    
    model = WikihopQAModel()
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-05, weight_decay=0.01, betas=(0.9, 0.98))
    epochs = 5
    batch_size = 1
    num_examples = len(train_dataset)
    training_steps = epochs * num_examples // batch_size
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=training_steps, power=1.0)
    
    def compute_accuracy(batch, outputs):
        _, batch_size, num_correct = outputs
        return {"accuracy": num_correct.item()}
        
    
    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_root_dir="./model/",
        optimizer=optimizer,
        scheduler=scheduler,
        compute_metrics=compute_accuracy,
        logger="wandb",
        log=True,
        max_epochs=epochs,
        use_mixed_precision=True,
        gradient_accumulation_steps=1, 
        warmup_steps=200,
        val_check_interval=10,
        project_name="WikihopQA",
    )
    
    trainer.train(model, train_loader, test_loader)
    