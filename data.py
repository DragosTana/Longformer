from torch.utils.data import Dataset
from transformers import AutoTokenizer, LongformerForMaskedLM
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from copy import copy
import re
import tqdm

class WikiDataset(Dataset):
    """
    Wikipedia dataset for pretraining. Loads the wikipedia dataset and tokenizes the text data.
    """
    def __init__(self,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_seq_len: int = 512,
                 num_workers: int = 16,
                 cache_dir: str = "./data", 
                 shuffle: bool = False,
                 n_docs: int = None,
                ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.n_docs = n_docs
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("Dataset loaded and tokenized!")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        labels = copy(self.data[idx]["input_ids"])
        return {"input_ids": self.data[idx]["input_ids"], "attention_mask": self.data[idx]["attention_mask"], "labels": labels}
        
    def split(self, split_ratio: float = 0.8):
        train_size = int(split_ratio * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size]) 
    
    def _process_data(self, data):
        """Tokenize the text data."""
        return self.tokenizer(["".join(x) for x in data["text"]])   
    
    def _group_texts(self, examples, max_seq_len=128, pad_token_id=1):
        """Group texts together in a dataset so that they match the block size.
        Add padding to the end of the texts if there are not enough."""
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        padding_length = 0
        # Pad the sequences to ensure uniform length.
        remainder = total_length % max_seq_len
        if remainder != 0:
            padding_length = max_seq_len - remainder
            for k, t in concatenated_examples.items():
                concatenated_examples[k] = t + [pad_token_id] * padding_length
            total_length += padding_length
            concatenated_examples["attention_mask"][-padding_length:] = [0] * padding_length
        # Split by chunks of max_seq_len.
        result = {
            k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    
    def _raw_text_to_tokens(self): 
        print("Loading wikipedia dataset...")
        raw_data = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=self.cache_dir, trust_remote_code=True, num_proc=self.num_workers)
        text_data = raw_data.remove_columns(['id', 'url', 'title',])
        text_data.shuffle() if self.shuffle else None
        #NOTE: Remove this line to load the entire dataset
        if self.n_docs is not None:
            text_data = text_data.select(range(self.n_docs)) 
        text_data = text_data.map(self._process_data, batched=True, num_proc=self.num_workers, remove_columns=text_data.column_names)
        dataset = text_data.map(self._group_texts, batched=True, num_proc=self.num_workers, remove_columns=text_data.column_names, fn_kwargs={"max_seq_len": self.max_seq_len, "pad_token_id": self.tokenizer.pad_token_id})

        return dataset
    
class IMDB(Dataset):
    def __init__(self,
             tokenizer_name: str = "distilbert/distilroberta-base",
             max_seq_len: int = 512,
             num_workers: int = 16,
             cache_dir: str = "./data", 
             shuffle: bool = False,
             longformer: bool = False,
             ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.longformer = longformer
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("IMDB dataset loaded and tokenized!")
        
    @staticmethod
    def clean_text(x):
        x = re.sub('<.*?>', ' ', x)
        x = re.sub('http\S+', ' ', x)
        x = re.sub('\s+', ' ', x)
        return x.lower().strip()

    def _preprocess_data(self, examples):
        examples["text"] = [self.clean_text(text) for text in examples["text"]]
        tokenized_data = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
        
        if self.longformer:
            attention_mask = torch.tensor(tokenized_data["attention_mask"])
            attention_mask[:, 0] = 2
            tokenized_data["attention_mask"] = attention_mask.tolist()
            
        return tokenized_data
    
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
    
class Hyperpartisan(Dataset):
    def __init__(self,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_seq_len: int = 512,
                 num_workers: int = 16,
                 cache_dir: str = "./data", 
                 shuffle: bool = False,
                 longformer: bool = False,
                ): 
        """
        Initializes the Hyperpartisan dataset.

        Args:
            tokenizer_name (str): Name of the tokenizer.
            max_seq_len (int): Maximum sequence length for tokenization.
            num_workers (int): Number of workers for data processing.
            cache_dir (str): Directory to cache the dataset.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.longformer = longformer
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("Hyperpartisan dataset loaded and tokenized!")
    
    @staticmethod
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
        
    def _preprocess_data(self, examples):
        examples["text"] = [self.clean_text(text) for text in examples["text"]]
        tokenized_examples = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
        tokenized_examples["labels"] = examples["hyperpartisan"]
        tokenized_examples["labels"] = [int(label) for label in tokenized_examples["labels"]]
        if self.longformer:
            attention_mask = torch.tensor(tokenized_examples["attention_mask"])
            attention_mask[:, 0] = 2
            tokenized_examples["attention_mask"] = attention_mask.tolist()
            
        return tokenized_examples
    
    def _raw_text_to_tokens(self):
        print("Loading Hyperpartisan News Detection dataset...")
        raw_data = load_dataset("SemEvalWorkshop/hyperpartisan_news_detection", "byarticle", cache_dir=self.cache_dir, trust_remote_code=True)
        raw_data = raw_data.remove_columns(['title', 'url', 'published_at'])
        tokenized_data = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["text", "hyperpartisan"])
        return tokenized_data["train"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def split(self, split_ratio: float = 0.8):
        train_size = int(split_ratio * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])
    
if __name__ == "__main__":
    
    data = Hyperpartisan(tokenizer_name="distilbert/distilroberta-base",
                            max_seq_len=2048, 
                            num_workers=16, 
                            cache_dir="./data", 
                            shuffle=True, 
                            longformer=True,
                            )
    print(data[0])