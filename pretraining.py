from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from copy import copy
import time

class WikiDataset(Dataset):
    """
    Wikipedia dataset for pretraining. Loads the wikipedia dataset and tokenizes the text data.
    """
    def __init__(self,
                 tokenizer_name: str = "roberta-base",
                 max_seq_len: int = 128,
                 num_workers: int = 16,
                 cache_dir: str = "./data", 
                 shuffle: bool = False
                ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        
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
    
    def _process_data(self, data, tokenizer):
        """Tokenize the text data."""
        return tokenizer(["".join(x) for x in data["text"]])   
    
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
        print(f"Loading wikipedia dataset...")
        raw_data = load_dataset("wikipedia", "20220301.en", split="train", cache_dir=self.cache_dir, trust_remote_code=True, num_proc=self.num_workers)
        text_data = raw_data.remove_columns(['id', 'url', 'title',])
        text_data.shuffle() if self.shuffle else None
        #NOTE: Remove this line to load the entire dataset
        #text_data = text_data.select(range(1000)) 
        text_data = text_data.map(self._process_data, batched=True, num_proc=self.num_workers, remove_columns=text_data.column_names, fn_kwargs={"tokenizer": self.tokenizer})
        dataset = text_data.map(self._group_texts, batched=True, num_proc=self.num_workers, remove_columns=text_data.column_names, fn_kwargs={"max_seq_len": self.max_seq_len, "pad_token_id": self.tokenizer.pad_token_id})

        return dataset
    
#dataset = WikiDataset(tokenizer_name="roberta-base", max_seq_len=128, num_workers=16, cache_dir="./data", shuffle=True)
#train, test = dataset.split()
#datacollator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15)
#train_loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=datacollator)
#test_loader = DataLoader(test, batch_size=8, shuffle=False, collate_fn=datacollator)
#
#for batch in train_loader:
#    print(batch)
#    break

BASE_PATH = '/model/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, history, scheduler=None, num_epochs=25, save_path='checkpoint', continue_training=False, start_epoch=0):
    # load trained model
    if continue_training:
        with open(BASE_PATH + 'weights/{}_{}.model'.format(save_path, start_epoch - 1), 'rb') as f:
            state = torch.load(f, map_location=DEVICE)
            model.load_state_dict(state)
        with open(BASE_PATH + 'weights/{}_{}.optimizer'.format(save_path, start_epoch - 1), 'rb') as f:
            state = torch.load(f, map_location=DEVICE)
            optimizer.load_state_dict(state)
        with open(BASE_PATH + 'weights/{}_{}.history'.format(save_path, start_epoch - 1), 'rb') as f:
            history = torch.load(f)
        if scheduler:
            with open(BASE_PATH + 'weights/{}_{}.scheduler'.format(save_path, start_epoch - 1), 'rb') as f:
                state = torch.load(f, map_location=DEVICE)
                scheduler.load_state_dict(state)

    for epoch in range(start_epoch, num_epochs):
        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_metrics = {}

            """Iterate over data.
            `dataloaders` is a dict{'train': train_dataloader
                                    'val': validation_dataloader}
            """
            iterator = tqdm(dataloaders[phase])
            for batch in iterator:
                """
                Batch comes as a dict.
                """
                for k in batch:
                    batch[k] = batch[k].to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(batch['src'],
                                    batch['dst'],
                                    batch['src_lengths'],
                                    batch['dst_lengths'])
                    _, preds = outputs.max(dim=2)

                    loss = criterion(outputs.view(-1, len(train_dataset.src_token2id)), batch['dst'].view(-1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                        optimizer.step()

                # statistics
                running_metrics.setdefault('loss', 0.0)
                running_metrics['loss'] += loss.item() * batch['src'].size(0)
                for pred, ground_truth in zip(preds, batch['dst']):
                    metrics = get_metrics(pred, ground_truth)       # supposed to return a dictionary of metrics
                    for metric_name in metrics:
                        running_metrics.setdefault(metric_name, 0.0)
                        running_metrics[metric_name] += metrics[metric_name]

            for metric_name in running_metrics:
                multiplier = 1
                average_metric = running_metrics[metric_name] / dataset_sizes[phase]
                history.setdefault(phase, {}).setdefault(metric_name, []).append(average_metric * multiplier)

            print('{} Loss: {:.4f} Rouge: {:.4f}'.format(
                phase, history[phase]['loss'][-1], history[phase]['rouge-l'][-1]))

            # LR scheduler
            if scheduler and phase == 'val':
                scheduler.step(history['val']['loss'][-1])

        # save model and history
        with open(BASE_PATH + 'weights/{}_{}.model'.format(save_path, epoch), 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(BASE_PATH + 'weights/{}_{}.optimizer'.format(save_path, epoch), 'wb') as f:
            torch.save(optimizer.state_dict(), f)
        with open(BASE_PATH + 'weights/{}_{}.history'.format(save_path, epoch), 'wb') as f:
            torch.save(history, f)
        if scheduler:
            with open(BASE_PATH + 'weights/{}_{}.scheduler'.format(save_path, epoch), 'wb') as f:
                torch.save(scheduler.state_dict(), f)


        time_elapsed = time.time() - since
        history.setdefault('times', []).append(time_elapsed)     # save times per-epoch
        print('Epoch {} complete in {:.0f}m {:.0f}s'.format(epoch, 
            time_elapsed // 60, time_elapsed % 60))
        print()


    
        



