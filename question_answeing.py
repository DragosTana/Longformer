import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel
from safetensors.torch import load_file

from src.models.longformer import Longformer
from src.models.config import LongformerConfig
from src.models.banded_gemm import pad_to_window_size
from src.utils.wikihop import WikihopQADataset
from src.trainer.trainer import Trainer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_longformer(model_path):
    config = LongformerConfig(
        vocab_size=50265,
        num_hidden_layers=6,
        hidden_size=768,
        num_attention_heads=12,
        max_position_embeddings=2050,
        attention_window=[256]*6,
        attention_dilation=[1]*6, 
        num_labels=2, 
        type_vocab_size=1,
    )
    longformer = Longformer(config, add_pooling_layer=False)
    state_dict = load_file(model_path)
    state_dict = {k.replace("roberta.", ""): v for k, v in state_dict.items()}
    longformer.load_state_dict(state_dict, strict=False)

    current_embed = longformer.embeddings.word_embeddings.weight
    current_vocab_size, embed_size = current_embed.size()
    new_embed = longformer.embeddings.word_embeddings.weight.new_empty(current_vocab_size + 4, embed_size)
    new_embed.normal_(mean=torch.mean(current_embed).item(), std=torch.std(current_embed).item())
    new_embed[:current_vocab_size] = current_embed
    longformer.embeddings.word_embeddings.num_embeddings = current_vocab_size + 4
    del longformer.embeddings.word_embeddings.weight
    longformer.embeddings.word_embeddings.weight = torch.nn.Parameter(new_embed)
    print("Loaded model")
    print(longformer)
    return longformer

def load_distilroberta(model_name='distilbert/distilroberta-base'):
    model = AutoModel.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
    distil_roberta = RobertaModel(config, add_pooling_layer=False)
    distil_roberta.load_state_dict(model.state_dict(), strict=False)
    del model
    model = distil_roberta
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
    candidate_len = candidate_ids.shape[1]
    support_len = support_ids.shape[1]

    if candidate_len + support_len <= max_seq_len:
        token_ids = torch.cat([candidate_ids, support_ids], dim=1)
        attention_mask = torch.ones(token_ids.shape, dtype=torch.long, device=token_ids.device)
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
            attention_mask[0, :candidate_len] = 2
            token_ids, attention_mask = pad_to_window_size(
                token_ids, attention_mask, model.config.attention_window[0], model.config.pad_token_id)

            activations = model(token_ids, attention_mask=attention_mask)[0]
            all_activations.append(activations)
            if end == truncate_seq_len:
                break

        return all_activations

def pad_to_max_length(input_ids, attention_mask, max_length, pad_token_id):
    padding_length = max_length - input_ids.size(1)
    assert padding_length == max_length - attention_mask.size(1)
    if padding_length > 0:
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)
    return input_ids, attention_mask

def get_activations_roberta(model, candidate_ids, support_ids, max_seq_len, truncate_seq_len):
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
    def __init__(self, longformer=False, model_path=None):
        super(WikihopQAModel, self).__init__()
        self.longformer = longformer    
        if longformer:
            self.model = load_longformer(model_path)
        else:
            self.model = load_distilroberta(model_path)
        self.answer_score = torch.nn.Linear(self.model.embeddings.word_embeddings.weight.shape[1], 1, bias=False)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self._truncate_seq_len = 100000000000
            
    def forward(self, data, return_predicted_index=False):
        candidate_ids, support_ids, prediction_indices, correct_prediction_index = data
        if self.longformer:
            activations = get_activations_longformer(
                self.model,
                candidate_ids,
                support_ids,
                2048,
                self._truncate_seq_len)
        else:
            activations = get_activations_roberta(
                self.model,
                candidate_ids,
                support_ids,
                512,
                self._truncate_seq_len)
        
        prediction_activations = [act.index_select(1, prediction_indices) for act in activations]
        prediction_scores = [
            self.answer_score(prediction_act).squeeze(-1)
            for prediction_act in prediction_activations
        ]
        average_prediction_scores = torch.cat(
            [pred_scores.unsqueeze(-1) for pred_scores in prediction_scores], dim=-1
        ).mean(dim=-1)
        loss = self.loss(average_prediction_scores, correct_prediction_index)
        batch_size = candidate_ids.new_ones(1) * prediction_activations[0].shape[0]
        predicted_answers = average_prediction_scores.argmax(dim=1)
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

def main(args):
    train_dataset = WikihopQADataset(args.train_data, shuffle_candidates=False, tokenize=True)
    val_dataset = WikihopQADataset(args.val_data, shuffle_candidates=False, tokenize=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=WikihopQADataset.collate_single_item)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=WikihopQADataset.collate_single_item)

    print(f"Length of train_loader: {len(train_loader)}")
    print(f"Length of test_loader: {len(test_loader)}")

    model = WikihopQAModel(longformer=args.longformer, model_path=args.model_path)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    epochs = args.epochs
    batch_size = 1
    num_examples = len(train_dataset)
    training_steps = epochs * num_examples // batch_size
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=training_steps, power=1.0)

    def compute_accuracy(batch, outputs):
        _, batch_size, num_correct = outputs
        return {"accuracy": num_correct.item()}

    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_root_dir=args.default_root_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_metrics=compute_accuracy,
        logger="wandb",
        log=False,
        max_epochs=epochs,
        use_mixed_precision=True,
        gradient_accumulation_steps=1, 
        warmup_steps=100,
        val_check_interval=args.val_check_interval,
        project_name=args.project_name,
    )

    trainer.train(model, train_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/wikihop/train.tokenized_2048.json")
    parser.add_argument("--val_data", type=str, default="data/wikihop/dev.tokenized_2048.json")
    parser.add_argument("--longformer", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="./checkpoint-3000/model.safetensors")
    parser.add_argument("--lr", type=float, default=3e-05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--default_root_dir", type=str, default="./model/")
    parser.add_argument("--val_check_interval", type=int, default=10)
    parser.add_argument("--project_name", type=str, default="WikihopQA")

    args = parser.parse_args()
    main(args)
