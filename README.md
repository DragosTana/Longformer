# Longformer: the long-document Transformer

![Transformers](/doc/imgs/dbl5lu1-528855a2-d961-4e5d-b7eb-b088db142382.jpg)

Despite the name, this aims to be yet another PyTorch implementation of a [Transformer](https://arxiv.org/abs/1706.03762) architecture. However, the focus is on implementing and reproducing at least some of the experiments from the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). Implementation of [BERT](https://arxiv.org/abs/1810.04805) and [RoBERTa](https://arxiv.org/abs/1907.11692) models was a necessary intermediate step.

I tried to follow the Hugginface [Transformer](https://github.com/huggingface/transformers) library idea of dividing the models from their config files. This helped develpment as well as readability and easy of use. To guarantee success and preserve my mental sanity I followed the instruction in [this](https://karpathy.github.io/2019/04/25/recipe/) blog post by our our lord and savior Andrej Karpathy.