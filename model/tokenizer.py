from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer

def main():
    
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer_roberta = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    TEXT = "Hello world!"
    
    tokens_bert = tokenizer_bert(
        (TEXT, TEXT),
        )
    
    tokens_roberta = tokenizer_roberta(
        [TEXT]
        )

    print("BERT tokens: ", tokens_bert)
    print("RoBERTa tokens: ", tokens_roberta)
    
    
    
main()