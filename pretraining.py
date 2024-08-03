import argparse
import logging
import os
import copy
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from safetensors.torch import load_file

from src.models.longformer import LongformerForMaskedLM
from src.models.longformer_attention import LongformerSelfAttention
from src.models.roberta import RobertaForMaskedLM
from src.trainer.trainer import Trainer
from src.utils.data import WikiText103

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)

def create_long_model(save_model_to, attention_window, max_pos):
    model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base", return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", model_max_length=max_pos)
    config = model.config   
    
    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers
    config.attention_mode = 'sliding_chunks'
    config.autoregressive = False
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def main(args):
    dataset = WikiText103(tokenizer_name=args.tokenizer_name,
                          max_seq_len=args.max_seq_len,
                          num_workers=args.num_workers,
                          cache_dir=args.cache_dir, 
                          shuffle=args.shuffle,
                          n_docs=args.n_docs)
    train, test, _ = dataset.split()

    datacollator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=datacollator, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=datacollator, shuffle=False)

    logger.info("Converting distilroberta to longformer")
    model, tokenizer = create_long_model(save_model_to=args.save_model_to, attention_window=args.attention_window, max_pos=args.max_pos)
    config = model.config
    model = RobertaLongForMaskedLM(config)
    state_dict = load_file(args.state_dict_path)
    model.load_state_dict(state_dict, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(args.save_model_to)
    logger.info("Model converted")
    print(model)

    epochs = args.epochs
    warmup_steps = args.warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=1.0)

    def compute_metrics(data, output):
        labels = data["labels"]
        loss, logits = output
        accuracy = (logits.argmax(-1) == labels).float().mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_root_dir=args.default_root_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_metrics=compute_metrics,
        logger="wandb",
        log=True,
        max_epochs=epochs,
        use_mixed_precision=False,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        warmup_steps=warmup_steps,
        val_check_interval=args.val_check_interval,
        project_name=args.project_name,
    )

    trainer.train(model, train_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="distilbert/distilroberta-base")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, default="./data")
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--n_docs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_model_to", type=str, default="./data/long_model")
    parser.add_argument("--attention_window", type=int, default=512)
    parser.add_argument("--max_pos", type=int, default=2048)
    parser.add_argument("--state_dict_path", type=str, default="./data/long_model/model.safetensors")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--default_root_dir", type=str, default="./model/")
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--project_name", type=str, default="Pretraining")

    args = parser.parse_args()
    main(args)
