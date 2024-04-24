from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from copy import copy

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
                 n_docs: int = 10000,
                ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.n_docs = n_docs
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
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
        text_data = text_data.select(range(self.n_docs)) 
        text_data = text_data.map(self._process_data, batched=True, num_proc=self.num_workers, remove_columns=text_data.column_names, fn_kwargs={"tokenizer": self.tokenizer})
        dataset = text_data.map(self._group_texts, batched=True, num_proc=self.num_workers, remove_columns=text_data.column_names, fn_kwargs={"max_seq_len": self.max_seq_len, "pad_token_id": self.tokenizer.pad_token_id})

        return dataset
    
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

class SQuAD(Dataset):
    """
    SQuAD dataset for question answering. Loads the SQuAD dataset from hugginface, tokenizes and preprocesses the data
    to make the input suitable for the model. Please reference the following link https://huggingface.co/datasets/rajpurkar/squad
    for more information about the dataset structure.
    
    The input to the model is a dictionary with the following keys:
    input_ids: tokenized input
    attention_mask: attention mask
    start_positions: start token position of the answer
    end_positions: end token position of the answer
    
    The input_ids is formatted as follows:
    [CLS] question [SEP] context [SEP]
    
    """
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
        print("SQuAD dataset loaded and tokenized!")
        
    def _preprocess_data(self, examples):
        questions = examples["question"]
        contexts = examples["context"]
        answers = examples["answers"]
        
        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_seq_len,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        
        return inputs
    
    def _raw_text_to_tokens(self):
        print("Loading SQuAD dataset...")
        raw_data = load_dataset("squad", cache_dir=self.cache_dir, trust_remote_code=True)
        # remove the columns that are not needed
        raw_data = raw_data.remove_columns(["id", "title"])
        tokenized_squad = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["question", "context", "answers"])
        return tokenized_squad
    
    def split(self):
        train = self.data["train"]
        val = self.data["validation"]
        return train, val
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    from transformers import DistilBertTokenizerFast
    from transformers import DefaultDataCollator
    qa = SQuAD(shuffle=True)
    train, val = qa.split()
    
    datacollator = DefaultDataCollator()
    dataloader = DataLoader(train, batch_size=1, collate_fn=datacollator)
    batch = next(iter(dataloader))
    
    #decode the batch
    print("Question and context: ")
    print(qa.tokenizer.decode(batch["input_ids"][0]))
    
    print("Answer: ")
    start = batch["start_positions"].item()
    end = batch["end_positions"].item()
    print(qa.tokenizer.decode(batch["input_ids"][0][start:end]))


