from transformers import DataCollatorWithPadding
from transformers import DistilBertForMaskedLM
from torch.utils.data import DataLoader
import torch
from model.distil_bert import MyDistiBertClassification
from model.config import ConfigClassification
from trainer import Trainer
from data import IMDB
    
dataset = IMDB(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
train, test = dataset.split()
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
train_loader = DataLoader(train.select(range(100)), batch_size=8, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test.select(range(100)), batch_size=8, shuffle=False, collate_fn=datacollator)

config = ConfigClassification(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522, num_labels=2)
model = MyDistiBertClassification(config)
try:
    model_state_dict = torch.load("./model/weights/distilbert.pth")
    model.load_state_dict(model_state_dict, strict=False)
except:
    # If the model is not found we will load the pretrained model from the Hugging Face library
    model_hf = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", cache_dir="./model/weights/")
    torch.save(model_hf.state_dict(), "./model/weights/distilbert.pth")
    model_state_dict = torch.load("./model/weights/distilbert.pth")
    model.load_state_dict(model_state_dict, strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def compute_metrics(data, outputs):
    predictions = torch.argmax(outputs, dim=-1)
    return {"accuracy": (predictions == data["labels"]).float().mean()}

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=compute_metrics,
    continue_training=False, 
    logger="wandb",
    log=True, 
    max_epochs=2,  
    gradient_accumulation_steps=2,
    project_name="Classification_IMDB",
)

trainer.fit(model, [train_loader], [test_loader])
