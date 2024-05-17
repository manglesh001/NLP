from preprocess import Preprocess, ElmoDataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ELMO(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, embedding_matrix):
        super(ELMO, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(device)

        self.lstm1 = nn.LSTM(embedding_dim, embedding_dim,
                             batch_first=True, bidirectional=False).to(device)
        self.lstm2 = nn.LSTM(embedding_dim, embedding_dim,
                             batch_first=True, bidirectional=False).to(device)
        
        self.linear1 = nn.Linear(embedding_dim, embedding_dim // 2).to(device)
        self.linear_out = nn.Linear(embedding_dim // 2, vocab_size).to(device)

    def forward(self, X):
        X = self.embedding(X)
        X, _ = self.lstm1(X)
        X, _ = self.lstm2(X)
        X = self.linear1(X)
        linear_out = self.linear_out(X)
        return linear_out
    
def train_model(forward_model, backward_model, train_dataloader, forward_optimizer, backward_optimizer, forward_criterion, backward_criterion, epochs):
    losses = {'epoch': [], 'train_loss': [], 'valid_loss': []}
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        forward_model.train()
        backward_model.train()
        total_loss = 0
        iter = 0
        for (forward, backward, label) in tqdm(train_dataloader, desc='Training'):
            forward = forward.to(device)
            backward = backward.to(device)
            forward_input_sequence = forward[:, :-1]
            forward_target_sequence = forward[:, 1:]
            backward_input_sequence = backward[:, :-1]
            backward_target_sequence = backward[:, 1:]

            forward_optimizer.zero_grad()
            backward_optimizer.zero_grad()
            forward_output = forward_model(forward_input_sequence)
            backward_output = backward_model(backward_input_sequence)
#             print(forward_output.shape,forward_target_sequence.shape)
            forward_loss = forward_criterion(forward_output.reshape(-1, VOCAB_SIZE), forward_target_sequence.reshape(-1))
#             print(forward_output.reshape(-1, VOCAB_SIZE).shape, forward_target_sequence.reshape(-1).shape)
            backward_loss = backward_criterion(backward_output.reshape(-1, VOCAB_SIZE), backward_target_sequence.reshape(-1))

            forward_loss.backward()
            backward_loss.backward()
            forward_optimizer.step()
            backward_optimizer.step()
            total_loss += forward_loss.item() + backward_loss.item()
            iter += 1
            if iter % 100 == 0:
                print('Iteration: ', iter, 'Train Loss: ', total_loss/iter)
        train_loss = total_loss / len(train_dataloader)
        print('Train Loss: ', train_loss)
        losses['epoch'].append(epoch)
        losses['train_loss'].append(train_loss)
    
    return losses


    

train_path = 'dataset/train.csv'
test_path = 'dataset/test.csv'
preprocessor = Preprocess(train_path,test_path)
train_sentences,word2idx=preprocessor.process_data('train')
print(len(train_sentences),len(word2idx))
test_sentences=preprocessor.process_data('test')
df1=pd.read_csv(train_path)
train_Y=df1['Class Index']

df2=pd.read_csv(test_path)
test_Y=df2['Class Index']

train_dataset = ElmoDataset(train_sentences,train_Y,word2idx)

test_dataset = ElmoDataset(test_sentences,test_Y,word2idx)


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = Word2Vec(sentences=train_sentences, vector_size=150, window=5, min_count=1, workers=4)

word_vectors = model.wv

vocab_size=len(word2idx)
EMBEDDING_DIM=150
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, idx in word2idx.items():
    if(word == '<PAD>'):
        embedding_matrix[idx] = np.zeros(EMBEDDING_DIM)
    else:
        embedding_matrix[idx] = word_vectors[word]
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

EMBEDDING_DIM=150
VOCAB_SIZE = vocab_size 
BATCH_SIZE=32
forward_model = ELMO(VOCAB_SIZE, EMBEDDING_DIM, BATCH_SIZE, embedding_matrix)
backward_model = ELMO(VOCAB_SIZE, EMBEDDING_DIM, BATCH_SIZE, embedding_matrix)

forward_optimizer = optim.Adam(forward_model.parameters(), lr=0.001)
backward_optimizer = optim.Adam(backward_model.parameters(), lr=0.001)

forward_criterion = nn.CrossEntropyLoss(ignore_index=0)
backward_criterion = nn.CrossEntropyLoss(ignore_index=0)

epochs=10
train_model(forward_model, backward_model, train_dataloader, forward_optimizer, backward_optimizer, forward_criterion, backward_criterion, epochs)
torch.save(forward_model.state_dict(), 'forward_model.pt')
torch.save(backward_model.state_dict(), 'backward_model.pt')

forward_embeddings = list(forward_model.parameters())[0].cpu().detach().numpy()
backward_embeddings = list(backward_model.parameters())[0].cpu().detach().numpy()

print(forward_embeddings.shape,backward_embeddings.shape)
