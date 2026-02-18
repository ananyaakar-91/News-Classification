# News-Classification

# Project Overview

This project fine-tunes RoBERTa-base for multi-class news classification on the AG News dataset.

The model classifies news articles into four categories:

 World

 Sports

 Business

 Sci/Tech

The goal was to build a high-performance Transformer-based text classifier using HuggingFace and PyTorch, with proper evaluation and training diagnostics.


# Model Architecture

Model: roberta-base

Framework: HuggingFace Transformers

Backend: PyTorch

Max Sequence Length: 128 tokens

Optimizer: AdamW

Learning Rate: 1e-5

Epochs: 4

Batch Size: 16

Early Stopping: Enabled

# Results
Metric	Score
Test Accuracy	95.25%
Macro F1 Score	0.9525
Training Epochs	4
Test Samples	7,600


The model shows strong overall generalization, with minor confusion between Business and Sci/Tech, which is expected due to semantic overlap.

# Training Diagnostics

The project includes:

Training vs Validation Loss curves

Validation Accuracy per epoch

Confusion Matrix visualization

Detailed classification report

Early stopping to prevent overfitting

# Tech Stack

Python

PyTorch

HuggingFace Transformers

HuggingFace Datasets

Scikit-Learn

Pandas & NumPy

Matplotlib & Seaborn

WordCloud (EDA)

# Dataset

Dataset: AG News Classification Dataset

Source: Kaggle

Classes: 4 balanced categories

Training Samples: 120,000

Test Samples: 7,600

# Project Workflow

Data loading & preprocessing

Exploratory Data Analysis (WordClouds, Bigram frequency, Class distribution)

Train/Validation split (stratified)

Tokenization using RoBERTa tokenizer

Fine-tuning RoBERTa

Evaluation (Accuracy + Macro F1)

Confusion Matrix & diagnostics
