import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


df = pd.read_csv('/kaggle/input/anlp-3/train.csv')

descriptions = df['Description'].iloc[0:15000]
def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [re.sub(r'[^a-zA-Z0-9 ]', '', word) for word in tokens]
    tokens = [word for word in tokens if word]
    return tokens


clean_descriptions = [preprocess(desc) for desc in descriptions if isinstance(desc, str)]

word_freq = Counter(word for sentence in clean_descriptions for word in sentence)

word_to_index = {'<UNK>': 0}
unk_count = 0  

for word, freq in word_freq.items():
    if freq >= 5:
        word_to_index[word] = len(word_to_index)
    else:
        unk_count += freq  


word_freq['<UNK>'] = unk_count

index_to_word = {index: word for word, index in word_to_index.items()}


sentences_as_indices = []
for sentence in clean_descriptions:
    sentence_indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
    sentences_as_indices.append(sentence_indices)
    
    
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)



    def forward(self, target_words, context_words):
        target_embeds = self.target_embeddings(target_words)
        context_embeds = self.context_embeddings(context_words)
        dots = torch.einsum('be,bce->bc', target_embeds, context_embeds)
        return dots
  


def generate_training_data(sentences, num_neg_samples, window_size, vocab_size):
    targets, contexts, labels = [], [], []
    pos_pairs = []
    context_dict = {}
    for sequence in sentences:
        for idx, word_idx in enumerate(sequence):
            context_dict[word_idx] = []
            window_start = max(0, idx - window_size)
            window_end = min(len(sequence), idx + window_size + 1)
            for context_index in range(window_start, window_end):
                if context_index != idx:
                    pos_pairs.append((word_idx, sequence[context_index]))
                    context_dict[word_idx].append(sequence[context_index])
    for center, context in pos_pairs:
        cntxt, lbl = [], []
        targets.append(center)
        cntxt.append(context)
        lbl.append(1)
        i = 0
        while i < num_neg_samples:
            negative_sample = np.random.randint(0, vocab_size)
            if negative_sample != center and negative_sample not in context_dict[center]:
                cntxt.append(negative_sample)
                lbl.append(0)
                i += 1
        contexts.append(cntxt)
        labels.append(lbl)
    return targets, contexts, labels

class SkipGramDataset(Dataset):
    def __init__(self, target_indices, context_indices, labels):
        self.target_indices = target_indices
        self.context_indices = context_indices
        self.labels = labels

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        return torch.tensor(self.target_indices[idx]), torch.tensor(self.context_indices[idx]), torch.tensor(self.labels[idx])
        
        
        

embedding_dim = 150  
vocab_size = len(word_to_index)  
window_size=1
learning_rate = 0.001
epochs = 10
batch_size = 300
num_neg_samples=2



target_indices, context_indices, labels = generate_training_data(sentences_as_indices, num_neg_samples, window_size, vocab_size)
print(len(target_indices), len(context_indices), len(labels))
dataset = SkipGramDataset(target_indices, context_indices, labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for target_word, context_word, label in data_loader:
        optimizer.zero_grad()
        scores = model(target_word, context_word)
        label_float = label.float()
        loss = criterion(scores, label_float)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss}')


embeddings_mean=(model.target_embeddings.weight + model.context_embeddings.weight)/2

model_save_name = 'skip-gram-word-vectors.pt'
path = F"/kaggle/working/{model_save_name}"
torch.save({'vocab': word_to_index, 'embeddings': embeddings_mean}, path)



import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

model_path = '/kaggle/input/modelpath/skip-gram-word-vectors.pt'
checkpoint = torch.load(model_path)
vocab = checkpoint['vocab']
embeddings = checkpoint['embeddings']

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
UNK_IDX = len(vocab)

df = pd.read_csv('/kaggle/input/anlp-3/train.csv')
descriptions = df['Description'].iloc[0:15000]
labels = df['Class Index'].iloc[0:15000]

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [re.sub(r'[^a-zA-Z0-9 ]', '', word) for word in tokens]
    tokens = [word for word in tokens if word]
    indices = [word_to_idx[token] if token in word_to_idx else word_to_idx['<UNK>'] for token in tokens]
    return indices

sentence_indices = []
for desc in descriptions:
    indices = preprocess(desc)
    sentence_indices.append(indices)



sentence_tensors = [torch.tensor(indices) for indices in sentence_indices]

padded_sequences = pad_sequence(sentence_tensors, batch_first=True, padding_value=0)


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



num_epochs = 5
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
torch.save(rnn_classifier.state_dict(), 'rnn_classifier_skipgram_30K.pt')


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


train_predictions = []
train_targets = []
with torch.no_grad():
    for tokens, labels in train_loader:
        tokens = tokens.long()
        logits = rnn_classifier(tokens)
        _, predicted = torch.max(logits, 1)
        train_predictions.extend(predicted.tolist())
        train_targets.extend(labels.argmax(dim=1).tolist())


test_predictions = []
test_targets = []
with torch.no_grad():
    for tokens, labels in test_loader:
        tokens = tokens.long()
        logits = rnn_classifier(tokens)
        _, predicted = torch.max(logits, 1)
        test_predictions.extend(predicted.tolist())
        test_targets.extend(labels.argmax(dim=1).tolist())


train_accuracy = accuracy_score(train_targets, train_predictions)
train_f1 = f1_score(train_targets, train_predictions, average='weighted')
train_precision = precision_score(train_targets, train_predictions, average='weighted')
train_recall = recall_score(train_targets, train_predictions, average='weighted')
train_conf_matrix = confusion_matrix(train_targets, train_predictions)

test_accuracy = accuracy_score(test_targets, test_predictions)
test_f1 = f1_score(test_targets, test_predictions, average='weighted')
test_precision = precision_score(test_targets, test_predictions, average='weighted')
test_recall = recall_score(test_targets, test_predictions, average='weighted')
test_conf_matrix = confusion_matrix(test_targets, test_predictions)


print("Train Set:")
print(f"Accuracy: {train_accuracy}")
print(f"F1 Score: {train_f1}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print("Confusion Matrix:")
print(train_conf_matrix)

print("\nTest Set:")
print(f"Accuracy: {test_accuracy}")
print(f"F1 Score: {test_f1}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print("Confusion Matrix:")
print(test_conf_matrix)



