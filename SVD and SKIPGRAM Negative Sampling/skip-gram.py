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
#       self.target = target
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
batch_size = 32
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


