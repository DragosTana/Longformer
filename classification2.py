from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModel, RobertaForSequenceClassification
from datasets import load_dataset
import torch
import re
import evaluate
import numpy as np

cache_dir = "./data"
max_seq_len = 512
longformer = False
model_path = "./model/weights"

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", model_max_length=512)
raw_data = load_dataset("SemEvalWorkshop/hyperpartisan_news_detection", "byarticle", cache_dir=cache_dir, trust_remote_code=True)
raw_data = raw_data.remove_columns(['title', 'url', 'published_at'])

def clean_text(text):
    cleaned_text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
    cleaned_text = re.sub(r"\n", " ", cleaned_text)
    cleaned_text = re.sub(r"&#160;", "", cleaned_text)
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
    
    #remove urls
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)
    cleaned_text = re.sub(r'www\S+', '', cleaned_text)
    cleaned_text = re.sub(r'href\S+', '', cleaned_text)
    
    #remove multiple spaces
    cleaned_text = re.sub(r"[ \s\t\n]+", " ", cleaned_text)
    
    #remove repetitions
    cleaned_text = re.sub(r"([!?.]){2,}", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", cleaned_text)
    
    return cleaned_text

def preprocess_data(examples):
    examples["text"] = [clean_text(text) for text in examples["text"]]
    tokenized_examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_len)
    tokenized_examples["labels"] = examples["hyperpartisan"]
    tokenized_examples["labels"] = [int(label) for label in tokenized_examples["labels"]]
    if longformer:
        attention_mask = torch.tensor(tokenized_examples["attention_mask"])
        attention_mask[:, 0] = 2
        tokenized_examples["attention_mask"] = attention_mask.tolist()
        
    return tokenized_examples

def split(dataset, split_ratio: float = 0.8):
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

data = raw_data.map(preprocess_data, batched=True, num_proc=16, remove_columns=["text", "hyperpartisan"])
data = data["train"]

train, test = split(data)

datacollator = DataCollatorWithPadding(tokenizer=tokenizer)

weights = AutoModel.from_pretrained("distilbert/distilroberta-base")
config = weights.config
config.num_labels = 2
model = RobertaForSequenceClassification(config)
model.load_state_dict(weights.state_dict(), strict=False)

print(model)
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

lr_scheduler_args = {"lr_end": 0, "power": 1.0}
epochs = 10
batch_size = 8
gradient_accumulation_steps = 2
total_steps = len(train) * epochs // (batch_size * gradient_accumulation_steps)
warmup_steps = 0.1 * total_steps

training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_dir=model_path,
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    run_name="hyperpartisan_news_detection",
    logging_steps=1,
    save_steps=32,
    eval_steps=32,
    save_total_limit=3,
    dataloader_num_workers=16,
    warmup_steps=warmup_steps,
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="polynomial",
    lr_scheduler_kwargs=lr_scheduler_args,
    #max_grad_norm=5.0,
    fp16=True,
    fp16_opt_level="O1",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=datacollator,
    train_dataset = train,
    eval_dataset = test,
    compute_metrics=compute_metrics,
)

trainer.train()
