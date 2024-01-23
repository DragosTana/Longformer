from transformers import BertTokenizer, BertModel
import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BERT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model_state.bin')
MAX_LEN = 128
print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print('Done.')
print(" ")

class SentimentClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(output['pooler_output'])
        return self.out(output)
    
model = SentimentClassifier(2)
model.load_state_dict(torch.load(BERT_CHECKPOINT))

model.eval()
model.to(device)

while True:
    review_text = input('Enter review: ')
    encoding = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        
    )
    
    input_ids = encoding['input_ids'].to(device)
    
    output = model(input_ids, encoding['attention_mask'].to(device))
    _, prediction = torch.max(output, dim=1)
    if prediction == 1:
        print('Positive review detected.')
    else:
        print('Negative review detected.')