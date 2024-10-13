# IMDb Sentiment Analysis with BERT

This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers) to classify movie reviews from the IMDb dataset. The model is built using PyTorch and the Hugging Face Transformers library.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Configuration](#configuration)
- [License](#license)

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

Make sure you have Python 3.6 or higher and the following libraries installed:

- torch
- transformers
- datasets

## Usage

To train the model, run the following command:

```bash
python src/backend/main.py
```

This will load the IMDb dataset, preprocess the data, train the model, and save the trained model to `trained_model.pth`.

## Model Architecture

The model consists of the following components:

- **Tokenizer**: BERT tokenizer is used to convert text into token IDs.
- **Classifier**: A custom classifier that includes:
  - An embedding layer
  - An encoder built with multiple encoder blocks
  - A feed-forward network
  - Softmax layer for output probabilities

## Training

The training process involves the following steps:

1. Load the IMDb dataset.
2. Tokenize the text data.
3. Create data loaders for training and testing.
4. Initialize the model, optimizer, and loss function.
5. Train the model for a specified number of epochs.

The training function computes the loss and updates the model weights using backpropagation.

## Testing

After training, the model can be tested on the test dataset to evaluate its accuracy. The testing function computes the accuracy based on the model's predictions.

## Configuration

The configuration settings for the model can be found in `config/config.py`. You can adjust the following parameters:

- `HEADS`: Number of attention heads in the multi-head self-attention mechanism.
- `HIDDEN_DIMS`: Dimensionality of the hidden layers.
- `NUM_ENCODER_BLOCKS`: Number of encoder blocks in the model.
- `NUM_CLASSES`: Number of output classes (sentiment categories).
- `BATCH_SIZE`: Size of the batches during training.
- `EPOCHS`: Number of training epochs.
- `LEARNING_RATE`: Learning rate for the optimizer.
