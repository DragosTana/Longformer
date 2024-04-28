from transformers import DataCollatorForLanguageModeling
from transformers import DistilBertForMaskedLM
from torch.utils.data import DataLoader
import torch
from model.distil_bert import MyDistilBertForMaskedLM
from model.config import Config
from trainer import Trainer
from data import WikiDataset    
    
dataset = WikiDataset(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
train, test = dataset.split()
datacollator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15)
train_loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=8, shuffle=False, collate_fn=datacollator)

config = Config(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522)
model = MyDistilBertForMaskedLM(config)
try:
    model_state_dict = torch.load("./model/weights/distilbert.pth")
    model.load_state_dict(model_state_dict, strict=False)
except:
    # If the model is not found we will load the pretrained model from the Hugging Face library
    model_hf = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", cache_dir="./model/weights/")
    torch.save(model_hf.state_dict(), "./model/weights/distilbert.pth")
    model_state_dict = torch.load("./model/weights/distilbert.pth")
    model.load_state_dict(model_state_dict, strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)   
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=None,
    continue_training=False, 
    logger="wandb",
    log=False, 
    max_epochs=1,  
    gradient_accumulation_steps=1,
    project_name="prova_wikipedia",
)

trainer.fit(model, [train_loader], [test_loader])






    
        



