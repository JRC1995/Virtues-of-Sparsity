## Virtues-of-Sparsity

Attempted implementation and experiment with [continuous sparsification](https://openreview.net/forum?id=BJe4oxHYPB) and [sparse representaions]
(https://arxiv.org/abs/1903.11257) (with k-winner activation along with duty cycles and everything)
with Transformers on Named Entity Recognition (CoNLL 2003).


Run [this](https://github.com/JRC1995/Virtues-of-Sparsity/blob/master/CoNLL2003/Train/Transformer_dynamic_sparse_train.py) for training with continuous sparsification.
Run [this](https://github.com/JRC1995/Virtues-of-Sparsity/blob/master/CoNLL2003/Train/Transformer_sparse_train.py) for training with sparse representaions with the method from [here](https://arxiv.org/abs/1903.11257)
Run [this] for training baseline Transformer.

As for results, baseline Transformer seems to still perform the best, but we get about the same result with continuous sparsification with halved parameters (continuous sparsification trims down parameter count to almost half the original)



