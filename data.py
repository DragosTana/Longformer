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
             tokenizer_name: str = "distilbert-base-uncased",
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
        tokenized_examples = self.tokenizer(examples["text"]) #, truncation=True, padding="max_length", max_length=self.max_seq_len)
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
    
        
    
class SQuAD(Dataset):
    """
    SQuAD dataset for question answering. Loads the SQuAD dataset from Huggingface, tokenizes and preprocesses the data
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
                 shuffle: bool = False):
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("SQuAD dataset loaded and tokenized!")
        
    def _preprocess_data(self, examples):
        questions = examples["question"]
        contexts = examples["context"]
        answers = examples["answers"]

        inputs = self.tokenizer(
            questions,
            contexts,
            #max_length=self.max_seq_len,
            #truncation="only_second",
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
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            context_end = context_start
            while context_end < len(sequence_ids) and sequence_ids[context_end] == 1:
                context_end += 1
            context_end -= 1

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
        # Remove the columns that are not needed
        raw_data = raw_data.remove_columns(["id", "title"])
        tokenized_squad = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["question", "context", "answers"])
        return tokenized_squad
    
    def split(self):
        train = self.data["train"]
        val = self.data["validation"]
        return train, val
    
    def __len__(self):
        return len(self.data["train"]) + len(self.data["validation"])
    
    def __getitem__(self, idx):
        if idx < len(self.data["train"]):
            return self.data["train"][idx]
        else:
            return self.data["validation"][idx - len(self.data["train"])]
    
    
def compare_answers(preprocessed_data, raw_data):
    for i in tqdm.tqdm(range(len(preprocessed_data))):
        preprocessed_answer = preprocessed_data[i]
        raw_answer = raw_data[i]
        
        preprocessed_text = raw_answer['context'][preprocessed_answer['start_positions']:preprocessed_answer['end_positions']]
        raw_text = raw_answer['answers']['text'][0]
        
        if preprocessed_text != raw_text:
            print("----------------------------------------------")
            print(f"Preprocessed answer: {preprocessed_text}")
            print(f"Raw answer: {raw_text}")
            print(f"Start position: {preprocessed_answer['start_positions']}")
            print(f"End position: {preprocessed_answer['end_positions']}")
            print(f"Real start position: {raw_answer['answers']['answer_start']}")
            print(f"Real end position: {raw_answer['answers']['answer_start'][0] + len(raw_text)}")
            
if __name__ == "__main__":
    
    from transformers import DataCollatorWithPadding
    from torch.utils.data import DataLoader    
    
    hyperpartisan = Hyperpartisan(tokenizer_name="distilbert/distilroberta-base",
                                  max_seq_len=512,
                                  num_workers=16,
                                  cache_dir="./data",
                                  shuffle=True,
                                  longformer=False)
    train, test = hyperpartisan.split()
    print(len(train), len(test))
    collator = DataCollatorWithPadding(tokenizer=hyperpartisan.tokenizer)
    train_loader = DataLoader(train, batch_size=4, shuffle=True, collate_fn=collator)
    example = next(iter(train_loader))
    print("Hyperpartisan dataset example:")
    #decode example
    print(hyperpartisan.tokenizer.decode(example["input_ids"][0]))
    
    
    
