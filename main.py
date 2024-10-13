import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from config.config import config
from Main.classifier import Classifier

# Load IMDb dataset
data = load_dataset('imdb')
train_data = data["train"]
test_data = data["test"]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization to the dataset
train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Create data loaders
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

train_data_loader = DataLoader(dataset=train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
test_data_loader = DataLoader(dataset=test_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn)

# Get the vocabulary size from the tokenizer
vocab_size = tokenizer.vocab_size

# Initialize model, optimizer, and loss function
model = Classifier(vocab_size, config["HIDDEN_DIMS"])
optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
loss_func = nn.CrossEntropyLoss()

# Training function
def train_model(model, train_loader, optimizer, loss_func, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Testing function
def test_model(model, test_loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")

# Train the model
train_model(model, train_data_loader, optimizer, loss_func, config["EPOCHS"])

# Save the model
torch.save(model.state_dict(), "trained_model.pth")

# Test the model
test_model(model, test_data_loader)
