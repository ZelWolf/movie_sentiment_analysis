# SentimentScope: Sentiment Analysis with a Transformer Built from Scratch

A PyTorch implementation of a transformer-based binary sentiment classifier, trained from scratch on the IMDB Large Movie Review Dataset. The project adapts a GPT-style decoder architecture for sequence classification and reaches **75.90% test accuracy** after three epochs of training.

---

## Table of Contents

1. [Overview](#overview)
2. [Results](#results)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training Configuration](#training-configuration)
6. [Repository Structure](#repository-structure)
7. [Getting Started](#getting-started)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [References](#references)

---

## Overview

The task is framed as binary classification: given a movie review, predict whether the sentiment is positive (1) or negative (0). The pipeline covers:

- Loading and exploring the IMDB dataset with Pandas
- Subword tokenization using the `bert-base-uncased` tokenizer (Hugging Face)
- A custom PyTorch `Dataset` and `DataLoader`
- A transformer (attention head, multi-head attention, feed-forward, block) implemented from scratch
- Two modifications for classification: **mean pooling** over token embeddings and a **linear classification head**
- Training with cross-entropy loss and the AdamW optimizer, with per-epoch validation

## Results

| Metric                         | Value  |
|--------------------------------|--------|
| Validation accuracy, epoch 1   | 70.84% |
| Validation accuracy, epoch 2   | 75.40% |
| Validation accuracy, epoch 3   | 77.00% |
| **Test accuracy (25,000 reviews)** | **75.90%** |

Training loss decreased from 0.6888 (first logged step) to 0.4138 (last logged step). An untrained model scores about 52% on validation, consistent with random chance on a balanced dataset.

## Dataset

[IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011).

| Split      | Samples | Notes                                   |
|------------|---------|-----------------------------------------|
| Train      | 22,500  | 90% of the 25,000 labelled training set |
| Validation | 2,500   | 10%, after shuffling (seed 42)          |
| Test       | 25,000  | Official test split                     |

- Classes are balanced (12,500 positive and 12,500 negative reviews in the original training set).
- Review length in the training set: mean 233.8 words, median 174, maximum 2,470.
- The `unsup/` folder is not used.

## Model Architecture

The model (`DemoGPT`) stacks transformer blocks over token and positional embeddings.

```
Token IDs (B, 128)
   -> Token Embedding + Positional Embedding        (B, 128, 128)
   -> 4 x Transformer Block
        [LayerNorm -> Multi-Head Attention -> residual]
        [LayerNorm -> Feed-Forward (GELU)  -> residual]
   -> Final LayerNorm
   -> Mean Pooling over sequence dimension          (B, 128)
   -> Linear Classification Head (no bias)          (B, 2)
```

| Hyperparameter     | Value                              |
|--------------------|------------------------------------|
| Vocabulary size    | 30,522 (`bert-base-uncased`)       |
| Embedding dim      | 128                                |
| Context length     | 128 tokens                         |
| Layers             | 4                                  |
| Attention heads    | 4 (head size 32)                   |
| Feed-forward width | 4 x 128 = 512                      |
| Dropout            | 0.1                                |
| Parameters         | approx. 4.7M                       |

### Adaptation for classification

1. **Mean pooling:** token-level outputs are averaged across the time dimension (`torch.mean(x, dim=1)`) to form one vector per review.
2. **Classification head:** a `Linear(d_embed, num_classes)` layer maps the pooled vector to two class logits.

## Training Configuration

| Setting        | Value                |
|----------------|----------------------|
| Loss           | Cross-entropy        |
| Optimizer      | AdamW                |
| Learning rate  | 3e-4                 |
| Batch size     | 32                   |
| Epochs         | 3                    |
| Max seq length | 128 (truncate / pad) |
| Hardware       | CUDA GPU             |

## Repository Structure

```
.
├── Sentiment Analysis using Transformer.ipynb   # Full pipeline: data, model, training, evaluation
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- PyTorch
- Hugging Face `transformers`
- pandas, numpy, matplotlib
- Jupyter
- A CUDA-capable GPU is recommended for training

### Installation

```bash
pip install torch transformers pandas numpy matplotlib jupyter
```

### Running

1. Download `aclImdb_v1.tar.gz` from the [dataset page](https://ai.stanford.edu/~amaas/data/sentiment/) and place it next to the notebook.
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "Sentiment Analysis using Transformer.ipynb"
   ```
3. Run all cells in order. The notebook extracts the archive, trains the model, and reports validation and test accuracy.

## Limitations and Future Work

- **Truncation:** inputs are cut at 128 tokens, while the median review is about 174 words. Longer context would retain more sentiment signal.
- **Causal attention mask:** the attention heads retain the causal (lower-triangular) mask inherited from the generative architecture. For classification, bidirectional attention would let every token attend to the full review and is likely to improve accuracy.
- **Padding in pooling:** mean pooling currently includes padded positions; masking them out would give cleaner representations.
- **Scale:** the model is small and trained for only 3 epochs. Larger embeddings, more layers, and longer training are natural next steps (validation accuracy was still rising at epoch 3).
- **Evaluation:** only accuracy is reported; precision, recall, F1, and a confusion matrix would give a fuller picture.

## References

- Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL (tokenizer).
