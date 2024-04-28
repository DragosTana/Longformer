from transformers import DistilBertForMaskedLM
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
import torch
from model.distil_bert import MyDistilBertForQuestionAnswering
from model.config import Config
from trainer import Trainer
from data import SQuAD

dataset = SQuAD(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
datacollator = DefaultDataCollator()
train, test = dataset.split()
train_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=datacollator)

config = Config(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522)
model = MyDistilBertForQuestionAnswering(config)
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

def calculate_f1_score(data, outputs):
    # Convert logits to predicted indices
    start_preds, end_preds = outputs
    start_preds = start_preds.argmax(dim=1)
    end_preds = end_preds.argmax(dim=1)
    
    # Calculate F1 score for each example
    f1_scores = []
    for start_pred, end_pred, start_target, end_target in zip(start_preds, end_preds, data["start_positions"], data["end_positions"]):
        start_pred = start_pred.item()
        end_pred = end_pred.item()
        start_target = start_target.item()
        end_target = end_target.item()
        pred_span = list(range(start_pred, end_pred + 1))
        true_span = list(range(start_target, end_target + 1))
        common = len(set(pred_span) & set(true_span))
        precision = common / max(1, len(pred_span))
        recall = common / max(1, len(true_span))
        f1 = (2 * precision * recall) / max(1e-9, precision + recall)
        f1_scores.append(f1)
    
    # Calculate average F1 score
    avg_f1_score = sum(f1_scores) / max(1, len(f1_scores))
    
    return {"f1_score": avg_f1_score}

trainer = Trainer(
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_root_dir="./model/",
    optimizer=optimizer,
    scheduler=scheduler,
    compute_metrics=calculate_f1_score,
    continue_training=False, 
    logger="wandb",
    log=True, 
    max_epochs=2,  
    gradient_accumulation_steps=1,
    project_name="QuestionAnswering_SQuAD",
)

trainer.fit(model, [train_loader], [test_loader])

#model_state_dict = torch.load("./model/weights/MyDistilBertForQuestionAnswering_final.model")
#model.load_state_dict(model_state_dict)
#
#example = next(iter(test_loader))
#model.eval()
#outputs = model(**example)
#start_preds, end_preds = outputs
#
#print("Input:")
#print(dataset.tokenizer.decode(example["input_ids"][0]))
#print("\n")
#print("Expected:")
#print(dataset.tokenizer.decode(example["input_ids"][0][example["start_positions"][0]:example["end_positions"][0]]))
#print("\n")
#print("Predicted:")
#print(dataset.tokenizer.decode(example["input_ids"][0][start_preds.argmax(dim=1).item():end_preds.argmax(dim=1).item()]))
#print("\n")
#print("F1 Score:")
#print(calculate_f1_score(example, outputs))
