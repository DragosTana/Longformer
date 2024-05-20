from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset   

class IMDB(Dataset):
    def __init__(self,
             tokenizer_name: str = "distilbert-base-uncased",
             max_seq_len: int = 512,
             num_workers: int = 16,
             cache_dir: str = "./data", 
             shuffle: bool = False,
             ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        self.data = self._raw_text_to_tokens()
        print("IMDB dataset loaded and tokenized!")
        
    def _preprocess_data(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
    
    def _raw_text_to_tokens(self):
        print("Loading IMDB dataset...")
        raw_data = load_dataset("imdb", cache_dir=self.cache_dir, trust_remote_code=True)
        tokenized_imdb = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["text"])
        return tokenized_imdb
    
    def split(self):
        return self.data["train"], self.data["test"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    