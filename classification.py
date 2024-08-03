from src.utils.data import IMDB, Hyperpartisan
from src.models.roberta import RobertaForSequenceClassification
from src.models.config import RobertaConfig
from src.models.longformer import LongformerForSequenceClassification
from src.models.config import LongformerConfig
from src.trainer.trainer import Trainer

from transformers import AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from safetensors.torch import load_file
import torch
import argparse

def compute_metrics(data, output):
    loss, logits = output
    preds = torch.argmax(logits, dim=1)
    return {"accuracy": (preds == data["labels"]).float().mean().item()}

def main(args):
    if args.dataset == "IMDB":
        data = IMDB()
    elif args.dataset == "Hyperpartisan":
        data = Hyperpartisan()
    else:
        raise ValueError("Dataset not supported")
        
    if args.model == "Roberta":
        config = RobertaConfig(vocab_size=50265,
                                num_hidden_layers=6,
                                hidden_size=768,
                                num_attention_heads=12,
                                max_position_embeddings=514,
                                num_labels=2, 
                                type_vocab_size=1,
                                )
        model = RobertaForSequenceClassification(config)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        distil_roberta = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
        model.load_state_dict(distil_roberta.state_dict(), strict=False)
    
    elif args.model == "Longformer":
        config = LongformerConfig(vocab_size=50265,
                                    num_hidden_layers=6,
                                    hidden_size=768,
                                    num_attention_heads=12,
                                    max_position_embeddings=2050,
                                    attention_window=[256]*6,
                                    attention_dilation=[1]*6, 
                                    num_labels=2, 
                                    type_vocab_size=1,
                                    )
            
        model = LongformerForSequenceClassification(config)
        state_dict = load_file(args.model_path)
        model.load_state_dict(state_dict, strict=False)
        
    train, test = data.split()
    datacollator = DataCollatorWithPadding(tokenizer=data.tokenizer)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=datacollator)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True, collate_fn=datacollator)

    epochs = args.epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = 0.1 * total_steps
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = PolynomialLR(optimizer, total_steps)
    
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
    parser.add_argument("--dataset", type=str, default="Hyperpartisan", choices=["IMDB", "Hyperpartisan"])
    parser.add_argument("--model", type=str, default="Roberta", choices=["Roberta", "Longformer"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--default_root_dir", type=str, default="./logs")
    parser.add_argument("--val_check_interval", type=float, default=200)
    parser.add_argument("--project_name", type=str, default="Classification")
    args = parser.parse_args()
    main(args)
