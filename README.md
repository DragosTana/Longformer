# Longformer: the long-document Transformer

![Transformers](/doc/imgs/dbl5lu1-528855a2-d961-4e5d-b7eb-b088db142382.jpg)

Despite the name, this aims to be yet another PyTorch implementation of a [Transformer](https://arxiv.org/abs/1706.03762) architecture. However, the focus is on implementing and reproducing at least some of the experiments from the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) reducing when needed (always) the dimensions of the datasets and of the models.

## Implementation note:

I used the Huggingface [Transformer](https://github.com/huggingface/transformers) library as a reference. When possible, however, I implemented things from scratch. Hugging Face was used just for:

- **I/O of the training data**: Pretraining requires big datasets, which are difficult and slow to handle on consumer grade machines. Huggingface provides the [Wikipedia](https://huggingface.co/datasets/wikipedia) dataset in a format that is already memory-mapped relieving me of the burden of having to implement this part from scratch.
- **Tokenizers**: Huggingface already has several implemented tokenizers, many of which are trained on the same dataset I use. Furthermore, the [fast-tokenizers](https://huggingface.co/learn/nlp-course/en/chapter6/3) are implemented in Rust making them much faster than anything I could have implemented from scratch in Python.
- **Pretrained models**: All models are implemented from scratch in PyTorch but are compatible with the weights of the pretrained models Hugging Face provides (at least [DistilBERTModel](/model/distil_bert.py/) is).


## Methodological note:

I was asked to implement a Longformer and reproduce at least some of the results from the paper