# Market Whisperer

A deep learning project for financial sentiment analysis using various neural network architectures to predict sentiment from financial text data.

## Overview

This project explores different deep learning approaches for analyzing sentiment in financial texts. The implementation includes comprehensive data preprocessing, exploratory data analysis, and evaluation of multiple model architectures.

## Implemented Methods

### CNN-Based Approaches
- **Conv1D**: Standard 1D convolutional neural network for text classification
- **TextCNN**: Multi-kernel CNN architecture capturing different n-gram patterns
- **Temporal Convolutional Network (TCN)**: Causal convolutions for sequential dependencies

### RNN-Based Approaches
- **Simple RNN**: Baseline recurrent neural network
- **LSTM**: Long Short-Term Memory networks
- **Bidirectional LSTM**: LSTM processing sequences in both directions
- **Bidirectional GRU**: Gated Recurrent Units with bidirectional processing
- **Multi-layered Bidirectional GRU**: Deep bidirectional GRU architecture

## Project Structure

```
market-whisperer/
├── fsa_data_preprocessing_and_eda.ipynb    # Data preprocessing and exploratory analysis
├── fsa_cnn_based_approach.ipynb            # CNN model implementations
└── fsa_rnn_based_approach.ipynb            # RNN model implementations
```

## Technologies

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: pandas, NLTK
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data Source**: Kaggle financial sentiment dataset

## Features

- Text preprocessing with lemmatization, stopword removal, and HTML tag cleaning
- Text vectorization with configurable vocabulary size and sequence length
- Model evaluation with AUC and accuracy metrics
- Experiment tracking and model versioning with wandb
- Early stopping and batch normalization for improved training

## Future Work

New methods and architectures will be evaluated in the future to further improve sentiment analysis performance.
