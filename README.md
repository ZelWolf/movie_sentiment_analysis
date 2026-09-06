# SentimentScope: Movie Review Sentiment Analysis

This repository contains my submission for Project 2 of the Udacity Machine Learning Nanodegree. The goal of this project was to train a transformer-based binary classification model from scratch to perform sentiment analysis on the IMDB dataset.

## Project Overview

While transformers are typically designed for generative tasks, this project adapts a generation-style transformer architecture for sequence classification. By applying mean pooling across the sequence dimension and passing the aggregated representation through a linear classification head, the model learns to classify movie reviews as positive or negative.

## Technical Highlights

* **Data Preprocessing:** Utilized the Hugging Face `bert-base-uncased` tokenizer to convert raw text into subword token IDs, providing an optimal balance between vocabulary size and semantic coverage.
* **Pipeline:** Built a custom PyTorch `Dataset` and `DataLoader` pipeline to manage padding, truncation (to a uniform sequence length), and batching.
* **Architecture:** Trained a customized transformer from scratch, condensing token-level embeddings via mean pooling before feeding them into a dense output layer.
* **Optimization:** Minimized cross-entropy loss over 3 epochs using the AdamW optimizer.

## Results & Performance

The model successfully surpassed the project's baseline requirement of 75% accuracy, demonstrating strong generalization to unseen text rather than simply memorizing the training data.

* **Initial Training Loss:** 0.6888
* **Final Training Loss:** 0.4138
* **Peak Validation Accuracy:** 77.00%
* **Final Test Accuracy:** **75.90%**

