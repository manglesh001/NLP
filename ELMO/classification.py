from preprocess import Preprocess, ElmoDataset
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ELMO import ELMO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class scoreClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fw_embeddings, bw_embeddings,
                 forward_lstm1, forward_lstm2, backward_lstm1, backward_lstm2, num_classes, requires_grad=True):
        super(scoreClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.fw_embeddings = nn.Embedding.from_pretrained(torch.tensor(fw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.fw_embeddings.weight.requires_grad = False 

        self.bw_embeddings = nn.Embedding.from_pretrained(torch.tensor(bw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.bw_embeddings.weight.requires_grad = True

        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33], dtype=torch.float).to(device), requires_grad=requires_grad)

        self.forward_lstm1 = forward_lstm1
        self.forward_lstm2 = forward_lstm2
        self.backward_lstm1 = backward_lstm1
        self.backward_lstm2 = backward_lstm2
        self.rnn = nn.LSTM(input_size=2 * embedding_dim, hidden_size=embedding_dim,
                           num_layers=2, batch_first=True, bidirectional=False).to(device)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim//2).to(device)
        self.linear_out = nn.Linear(embedding_dim//2, num_classes).to(device)  
#         self.linear1 = nn.Linear(2 * embedding_dim, embedding_dim).to(device)
#         self.linear_out = nn.Linear(embedding_dim, num_classes).to(device)

    def forward(self, forward, backward):
        fw_embed = self.fw_embeddings(forward)
        bw_embed = self.bw_embeddings(backward)

        fw_lstm1, _ = self.forward_lstm1(fw_embed)
        fw_lstm2, _ = self.forward_lstm2(fw_lstm1)
        bw_lstm1, _ = self.backward_lstm1(bw_embed)
        bw_lstm2, _ = self.backward_lstm2(bw_lstm1)

        embed = torch.cat((fw_embed, bw_embed), dim=2)
        lstm1 = torch.cat((fw_lstm1, bw_lstm1), dim=2)
        lstm2 = torch.cat((fw_lstm2, bw_lstm2), dim=2)

        elmo_out = self.weights[0] * embed + self.weights[1] * lstm1 + self.weights[2] * lstm2
        elmo_max = torch.max(elmo_out, dim=1)[0]
        output,_ = self.rnn(elmo_max)
        output = self.linear1(output)
#         output, _ = self.rnn(output)
        output = self.linear_out(output)
#         lin_out1 = self.linear1(elmo_max)
#         linear_out = self.linear_out(lin_out1)

        return output
    
class scoreClassifierWithLinearFunction(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fw_embeddings, bw_embeddings,
                 forward_lstm1, forward_lstm2, backward_lstm1, backward_lstm2, num_classes, requires_grad=True):
        super(scoreClassifierWithLinearFunction, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.fw_embeddings = nn.Embedding.from_pretrained(torch.tensor(fw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.fw_embeddings.weight.requires_grad = False 

        self.bw_embeddings = nn.Embedding.from_pretrained(torch.tensor(bw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.bw_embeddings.weight.requires_grad = True

#         self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33], dtype=torch.float).to(device), requires_grad=requires_grad)
        self.forward_lstm1 = forward_lstm1
        self.forward_lstm2 = forward_lstm2
        self.backward_lstm1 = backward_lstm1
        self.backward_lstm2 = backward_lstm2
        self.linear_function = nn.Linear(3*2*embedding_dim,2*embedding_dim).to(device)
        self.rnn = nn.LSTM(input_size=2 * embedding_dim, hidden_size=embedding_dim,
                           num_layers=2, batch_first=True, bidirectional=False).to(device)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim//2).to(device)
        self.linear_out = nn.Linear(embedding_dim//2, num_classes).to(device)  
#         self.linear1 = nn.Linear(2 * embedding_dim, embedding_dim).to(device)
#         self.linear_out = nn.Linear(embedding_dim, num_classes).to(device)

    def forward(self, forward, backward):
        fw_embed = self.fw_embeddings(forward)
        bw_embed = self.bw_embeddings(backward)

        fw_lstm1, _ = self.forward_lstm1(fw_embed)
        fw_lstm2, _ = self.forward_lstm2(fw_lstm1)
        bw_lstm1, _ = self.backward_lstm1(bw_embed)
        bw_lstm2, _ = self.backward_lstm2(bw_lstm1)

        embed = torch.cat((fw_embed, bw_embed), dim=2)
        lstm1 = torch.cat((fw_lstm1, bw_lstm1), dim=2)
        lstm2 = torch.cat((fw_lstm2, bw_lstm2), dim=2)
        elmo_out = self.linear_function(torch.cat((embed,lstm1,lstm2),dim=2))
#         elmo_out = self.weights[0] * embed + self.weights[1] * lstm1 + self.weights[2] * lstm2
        elmo_max = torch.max(elmo_out, dim=1)[0]
        output,_ = self.rnn(elmo_max)
        output = self.linear1(output)
#         output, _ = self.rnn(output)
        output = self.linear_out(output)
#         lin_out1 = self.linear1(elmo_max)
#         linear_out = self.linear_out(lin_out1)

        return output
    
def plot_confusion_matrix(true_labels,predictions,file_name):
  cm = confusion_matrix(true_labels, predictions)
  plt.figure(figsize=(10, 7))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.title('Confusion Matrix')
  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  plt.savefig(f'/kaggle/working/{file_name}')
  plt.show()


def print_evaluation_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    print("Accuracy:", accuracy)

    precision = precision_score(true_labels, predictions, average='weighted')
    print("Precision:", precision)

    recall = recall_score(true_labels, predictions, average='weighted')
    print("Recall:", recall)

    f1 = f1_score(true_labels, predictions, average='weighted')
    print("F1 Score:", f1)

def train_classifier(model, train_dataloader, optimizer, criterion, device, epochs=1):
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        train_loss = 0
        iter = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            forward, backward, labels = batch
            logits = model(forward, backward)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(logits, dim=1)
            train_loss += loss.item()
            iter += 1
            if iter % 100 == 0:
                print("Iteration: {}, Train Loss: {}".format(iter, loss.item()))

def test_model(model, test_loader, criterion, split, file_name):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    true_labels, predictions = [], []
    with torch.no_grad():  # Disable gradient calculation
        for inputs, backward, labels in test_loader:
            # Move data to the same device as the model
            inputs = inputs.to(device)
            backward = backward.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, backward)
            loss = criterion(outputs, labels)

            # Calculate loss
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_labels = labels.argmax(dim=1)  # Ensure this matches label format
            true_labels.extend(correct_labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            correct_predictions += (predicted == correct_labels).sum().item()

            total_samples += labels.size(0)

    # Compute metrics
    average_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    print(f"{split} Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    print_evaluation_metrics(true_labels, predictions)
    plot_confusion_matrix(true_labels, predictions, f"elmo_with_{file_name}_{split}")


train_path = '/kaggle/input/dataset/train.csv'
test_path = '/kaggle/input/dataset/test.csv'
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

vocab_size=len(word2idx)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

forward_model = torch.load('forward_model.pt')
backward_model = torch.load('backward_model.pt')

forward_embeddings = list(forward_model.parameters())[0].cpu().detach().numpy()
backward_embeddings = list(backward_model.parameters())[0].cpu().detach().numpy()

print(forward_embeddings.shape,backward_embeddings.shape)


EMBEDDING_DIM=150
VOCAB_SIZE = vocab_size 
BATCH_SIZE=32
num_classes=4
classifier1 = scoreClassifier(VOCAB_SIZE, EMBEDDING_DIM, forward_embeddings,backward_embeddings,
                 forward_model.lstm1, forward_model.lstm2, backward_model.lstm1, backward_model.lstm1, num_classes,requires_grad=False)

classifier1.to(device)
optimizer1 = optim.Adam(classifier1.parameters(), lr=0.001)
criterion1 = nn.CrossEntropyLoss()


classifier2 = scoreClassifier(VOCAB_SIZE, EMBEDDING_DIM, forward_embeddings,backward_embeddings,
                 forward_model.lstm1, forward_model.lstm2, backward_model.lstm1, backward_model.lstm1, num_classes,requires_grad=True)

classifier2.to(device)
#Initializing optimizer
optimizer2 = optim.Adam(classifier2.parameters(), lr=0.001)
#Initializing loss function
criterion2 = nn.CrossEntropyLoss()

train_classifier(classifier2, train_dataloader, optimizer2, criterion2, device, epochs=3)
print(classifier2.weights)
test_model(classifier2, train_dataloader, criterion2,"train","unfrozen_weights")
test_model(classifier2, test_dataloader, criterion2,"test","unfrozen_weights")

torch.save(classifier1.state_dict(), 'classifier1_frozen_weigths.pt')
torch.save(classifier2.state_dict(), 'classifier2_unfrozen_weights.pt')


classifier3 = scoreClassifierWithLinearFunction(VOCAB_SIZE, EMBEDDING_DIM, forward_embeddings,backward_embeddings,
                 forward_model.lstm1, forward_model.lstm2, backward_model.lstm1, backward_model.lstm1, num_classes)

print(classifier3)
classifier3.to(device)
#Initializing optimizer
optimizer3 = optim.Adam(classifier2.parameters(), lr=0.001)
#Initializing loss function
criterion3 = nn.CrossEntropyLoss()

train_classifier(classifier3, train_dataloader, optimizer3, criterion3, device, epochs=5)
test_model(classifier3, train_dataloader, criterion3,"train","Linear_function")
test_model(classifier3, test_dataloader, criterion3,"test","Linear_function")

