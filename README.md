# Longformer: the long-document Transformer

## Pretrainnig
This script is designed for pre-training a Long-RoBERTa model. It supports the WikiText103 dataset for language modeling tasks and allows for flexible configuration of various hyperparameters.

`python pretriaing.py --tokenizer_name <TOKENIZER_NAME> [other optional arguments]`

- --tokenizer_name: The name of the tokenizer to use. Default is "distilbert/distilroberta-base".
- --max_seq_len: The maximum sequence length for tokenization. Default is 2048.
- --num_workers: The number of workers for data loading. Default is 16.
- --cache_dir: Directory to cache the dataset. Default is "./data".
- --batch_size: The batch size for training and evaluation. Default is 1.
- --save_model_to: Directory to save the converted model. Default is "./data/long_model".
- --attention_window: The attention window size for Longformer. Default is 512.
- --max_pos: The maximum position embeddings for Longformer. Default is 2048.
- --state_dict_path: Path to the state dictionary file. Default is "./data/long_model/model.safetensors".
- --epochs: The number of training epochs. Default is 10.
- --warmup_steps: The number of warmup steps for learning rate scheduler. Default is 500.
- --gradient_accumulation_steps: The number of steps to accumulate gradients before updating the model parameters. Default is 8.
- --lr: The learning rate for the optimizer. Default is 5e-5.
- --default_root_dir: The root directory for saving logs and checkpoints. Default is "./model/".
- --val_check_interval: The interval at which to validate the model during training. Default is 500.
- --project_name: The project name for logging purposes (e.g., using Weights and Biases). Default is "Pretraining".

## Classification
This script allows for training and evaluating text classification models using either the RoBERTa or Longformer architectures on specified datasets. The primary datasets supported are IMDB and Hyperpartisan, which cater to sentiment analysis and hyperpartisan news detection tasks, respectively.
To run:

`python3 classification.py --dataset <DATASET> --model <MODEL> [other optional arguments]`

- --dataset: The dataset to use for training. Choices are "IMDB" or "Hyperpartisan". Default is "Hyperpartisan".
- --model: The model architecture to use. Choices are "Roberta" or "Longformer". Default is "Roberta".
- --model_path: Path to the model weights file. This is required if using the Longformer model. Default is None.
- --batch_size: The batch size for training and evaluation. Default is 8.
- -lr: The learning rate for the optimizer. Default is 5e-5.
- --epochs: The number of training epochs. Default is 3.
- --gradient_accumulation_steps: The number of steps to accumulate gradients before updating the model parameters. Default is 1.
- --default_root_dir: The root directory for saving logs and checkpoints. Default is "./logs".
- --val_check_interval: The interval at which to validate the model during training. Default is 200.
- --project_name: The project name for logging purposes (e.g., using Weights and Biases). Default is "Classification".

### Note
- For Longformer, you must specify the --model_path argument with the path to the model's weight file. If you don't have one please use the pretrain script first
- Logs and checkpoints will be saved in the directory specified by --default_root_dir.
- The script uses the Weights and Biases (wandb) library for logging. Ensure you have an account and are logged in to use this feature effectively.

## Question Answering

This script is designed for training and evaluating a model for the WikiHop QA task. It supports both RoBERTa and Longformer architectures, providing flexibility for handling long documents.

`python <script_name>.py --train_data <TRAIN_DATA> --val_data <VAL_DATA> [other optional arguments]`

- --train_data: Path to the training data file. Default is "data/wikihop/train.tokenized_2048.json".
- --val_data: Path to the validation data file. Default is "data/wikihop/dev.tokenized_2048.json".
- --longformer: Boolean flag to indicate if Longformer should be used. Default is False.
- --model_path: Path to the model weights file. Default is "./checkpoint-3000/model.safetensors".
- --lr: The learning rate for the optimizer. Default is 3e-5.
- --weight_decay: The weight decay for the optimizer. Default is 0.01.
- --epochs: The number of training epochs. Default is 5.
- --default_root_dir: The root directory for saving logs and checkpoints. Default is "./model/".
- --val_check_interval: The interval at which to validate the model during training. Default is 10.
- --project_name: The project name for logging purposes. Default is "WikihopQA".

### Note
- Here the dataset will not automatically download if it is not present on your machine. Please download it from the following [link](https://zenodo.org/records/6407402).
