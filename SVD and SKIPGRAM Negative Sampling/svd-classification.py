import pandas as pd

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import string
import re
import torch

df = pd.read_csv('/kaggle/input/train-csv/train.csv')
descriptions = df['Description'].iloc[0:15000]
labels = df['Class Index'].iloc[0:15000]

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [re.sub(r'[^a-zA-Z0-9 ]', '', word) for word in tokens]
    tokens = [word for word in tokens if word]
    indices = [word_to_idx[token] if token in word_to_idx else word_to_idx["<UNK>"] for token in tokens]
    return indices

sentence_indices = []
for desc in descriptions:
    indices = preprocess(desc)
    sentence_indices.append(indices)
sentence_tensors = [torch.tensor(indices) for indices in sentence_indices]
padded_sequences = pad_sequence(sentence_tensors, batch_first=True, padding_value=0)




model_path = '/kaggle/working/svd-word-vectors_15k.pt'
checkpoint = torch.load(model_path)
vocab = checkpoint['vocab']
embeddings = checkpoint['embeddings']

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
UNK_IDX = len(vocab)


class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_hot_target = np.zeros(4)
        one_hot_target[self.y[idx]-1] = 1
        return self.X[idx], torch.tensor(one_hot_target, dtype=torch.float32)


train_dataset = NewsDataset(padded_sequences, labels)

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float32))
        self.rnn = nn.LSTM(input_size, hidden_size,num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output[:, -1, :])
        return logits

input_size = embeddings.shape[1]
hidden_size = 300
output_size = 4 
rnn_classifier = RNNClassifier(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    rnn_classifier.train()
    for tokens, labels in train_loader:
        optimizer.zero_grad()
        tokens = tokens.long()
        logits = rnn_classifier(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

torch.save(rnn_classifier.state_dict(), 'rnn_classifier_15K.pt')



test_df = pd.read_csv('/kaggle/input/anlp-3/test.csv')
test_descriptions = test_df['Description']
test_labels = test_df['Class Index']

test_sentence_indices = []
for desc in test_descriptions:
    indices = preprocess(desc)
    test_sentence_indices.append(indices)
test_sentence_tensors = [torch.tensor(indices) for indices in test_sentence_indices]
padded_test_sequences = pad_sequence(test_sentence_tensors, batch_first=True, padding_value=0)

test_dataset = NewsDataset(padded_test_sequences, test_labels)
test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False)

rnn_classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for tokens, labels in test_loader:
        tokens = tokens.long()
        logits = rnn_classifier(tokens)
        #print(logits.shape)
        #print(labels.shape)
        _, predicted = torch.max(logits, 1)
        #print(predicted.shape)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy}")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix



test_predictions = []
test_targets = []
with torch.no_grad():
    for tokens, labels in test_loader:
        tokens = tokens.long()
        logits = rnn_classifier(tokens)
        _, predicted = torch.max(logits, 1)
        test_predictions.extend(predicted.tolist())
        test_targets.extend(labels.argmax(dim=1).tolist())

test_accuracy = accuracy_score(test_targets, test_predictions)
test_f1 = f1_score(test_targets, test_predictions, average='weighted')
test_precision = precision_score(test_targets, test_predictions, average='weighted')
test_recall = recall_score(test_targets, test_predictions, average='weighted')
test_conf_matrix = confusion_matrix(test_targets, test_predictions)

print("\nTest Set:")
print(f"Accuracy: {test_accuracy}")
print(f"F1 Score: {test_f1}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print("Confusion Matrix:")
print(test_conf_matrix)


