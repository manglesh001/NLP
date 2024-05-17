#FFNN TAGGER

import torch
import nltk
nltk.download('punkt')
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from conllu import parse
from gensim.models import Word2Vec
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Define paths to your CoNLL-U files
train_path = "/content/drive/MyDrive/ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-train.conllu"
test_path = "/content/drive/MyDrive/ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-test.conllu"
dev_path = "/content/drive/MyDrive/ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-dev.conllu"


def load_conllu(file_path, p, s):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    sentences = []
    pos_tags = []
    for sentence in parse(data):
        words = [token["form"].lower() for token in sentence]
        tags = [token["upos"] for token in sentence]

        # Add <START> and <POS> tags
        for i in range(0,p):

          words.insert(0,'<START>')
          words.append('<END>')

        for j in range(0,s):

          tags.insert(0,'<POS>')
          tags.append('<POS>')

        sentences.append(words)
        pos_tags.append(tags)
    #print(sentences[0], pos_tags[0])
    return sentences, pos_tags
# Load and preprocess data
p=3
s=3
train_sentences, train_pos_tags = load_conllu(train_path,p,s)
#print(len(train_sentences[0]))
# print(train_pos_tags)
dev_sentences,dev_pos_tags=load_conllu(dev_path,p,s)
test_sentences, test_pos_tags = load_conllu(test_path,p,s)


# # Build vocabulary
word_counts = Counter(word for sentence in train_sentences for word in sentence)
word_vocab = ["<UNK>"] + [word for word, count in word_counts.items() if count > 1]
#print(len(train_sentences[0]))
# print(wordcounts)
# print(word_vocab)_
for i, sentence in enumerate(train_sentences):
    train_sentences[i] = ['<UNK>' if word_counts[word] == 1 else word for word in sentence]
#print(train_sentences)


def replace_oov_with_unk(sentences, vocab):
    # Replace out-of-vocabulary words with <UNK> tag
    for i, sentence in enumerate(sentences):
        sentences[i] = ['<UNK>' if word not in vocab else word for word in sentence]
    return sentences

# Preprocess dev_sentences and test_sentences
dev_sentences = replace_oov_with_unk(dev_sentences, word_vocab)
test_sentences = replace_oov_with_unk(test_sentences, word_vocab)
# print(dev_sentences[0])
# print(test_sentences[0])

X_train = []
y_train = []

# Iterate through each sentence in train_sentences
for id,sentence in enumerate(train_sentences):
    # Iterate through the words in the sentence
    for i in range(len(sentence)):
        # Extract window of size 7 centered around the 3rd index word
        if i >= 0 and i < len(sentence) - 1:   #p-3 s+1 -4
            window = sentence[i - 0:i + 1]  # Extract window of size 7
            target_pos_tag = train_pos_tags[id][i]  # Get the POS tag of the word at the 3rd index
            X_train.append(window)
            y_train.append(target_pos_tag)

# Make a list of tuples containing (X, y) pairs
data = list(zip(X_train, y_train))
# print(len(X_train[0]))
# print(X_train[0])
# print(y_train[0])



X_dev = []
y_dev = []

# Iterate through each sentence in dev_sentences
for id,sentence in enumerate(dev_sentences):
    # Iterate through the words in the sentence
    for i in range(len(sentence)):
        # Extract window of size 7 centered around the 3rd index word
        if i >= 0 and i < len(sentence) - 1:   #p-3 s+1 -4
            window = sentence[i - 0:i + 1]  # Extract window of size 7
            target_pos_tag = dev_pos_tags[id][i]  # Get the POS tag of the word at the 3rd index
            X_dev.append(window)
            y_dev.append(target_pos_tag)

# Make a list of tuples containing (X, y) pairs
data = list(zip(X_dev, y_dev))
# print(len(X_dev[0]))
# print(X_dev[0])
# print(y_dev[0])

X_test = []
y_test = []

# Iterate through each sentence in test_sentences
for id,sentence in enumerate(test_sentences):
    # Iterate through the words in the sentence
    for i in range(len(sentence)):
        # Extract window of size 7 centered around the 3rd index word
        if i >= 0 and i < len(sentence) - 1:   #p-3 s+1 -4
            window = sentence[i - 0:i + 1]  # Extract window of size 7
            target_pos_tag = test_pos_tags[id][i]  # Get the POS tag of the word at the 3rd index
            X_test.append(window)
            y_test.append(target_pos_tag)

# Make a list of tuples containing (X, y) pairs
data = list(zip(X_test, y_test))
# print(len(X_test[0]))
# print(X_test[0])
# print(y_test[0])


# Train Word2Vec model
word2vec_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, workers=4)


# Convert each word to its corresponding embedding TRAIN
word_embeddings_train= []
for sentence in X_train:
    sentence_embeddings = [word2vec_model.wv[word] if word in word2vec_model.wv else [0] * 100 for word in sentence]
    word_embeddings_train.append(sentence_embeddings)

print(len(word_embeddings_train[0]))
# print(word_embeddings_train[0])


# Convert each word to its corresponding embedding DEV
word_embeddings_dev= []
for sentence in X_dev:
    sentence_embeddings = [word2vec_model.wv[word] if word in word2vec_model.wv else [0] * 100 for word in sentence]
    word_embeddings_dev.append(sentence_embeddings)

print(len(word_embeddings_dev[0]))
# print(word_embeddings_dev[0])

# Convert each word to its corresponding embedding TEST
word_embeddings_test= []
for sentence in X_test:
    sentence_embeddings = [word2vec_model.wv[word] if word in word2vec_model.wv else [0] * 100 for word in sentence]
    word_embeddings_test.append(sentence_embeddings)

print(len(word_embeddings_test[0]))
# print(word_embeddings_test[0])

# Create POS tag vocabulary (pos_vocab)
pos_vocab = defaultdict(lambda: len(pos_vocab))
for tags in train_pos_tags:
    for tag in tags:
        pos_vocab[tag]

print(pos_vocab)

reverse_pos_vocab = {}

# Iterate over the key-value pairs in pos_vocab
for key, value in pos_vocab.items():
    reverse_pos_vocab[value] = key

print(reverse_pos_vocab)

# Convert pos_vocab to lists of POS tags and their corresponding indices
pos_tags, pos_indices = zip(*pos_vocab.items())

# Create an instance of LabelBinarizer
label_binarizer = LabelBinarizer()

# Fit the LabelBinarizer to the POS tags
label_binarizer.fit(pos_tags)

# Transform y using the fitted LabelBinarizer to obtain one-hot encodings
y_one_hot_train = label_binarizer.transform(y_train)
y_one_hot_dev = label_binarizer.transform(y_dev)
y_one_hot_test = label_binarizer.transform(y_test)

# print(y_one_hot_train)
# print(y_one_hot_dev)
# print(y_one_hot_test)

START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNKNOWN_TOKEN="<UNK>"
PAD_TOKEN = "<pad>"

class EntityDataset(Dataset):
    def __init__(self, data: list[tuple[list[str], list[int]]], vocabulary: Vocab|None = None):
        """Initialize the dataset. Setup Code goes here"""
        self.sentences = [i[0] for i in data]  # list of sentences
        self.labels = [i[1] for i in data]
        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])  # use min_freq for handling unknown words better
            # self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            # if vocabulary provided use that
            self.vocabulary = vocabulary

    def __len__(self) -> int:
        """Returns number of datapoints."""
        return len(self.sentences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the datapoint at index."""
        return torch.tensor(self.sentences[index]), torch.tensor(self.labels[index]).float()

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together"""
        sentences = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])  # pad sentences with pad token id
        padded_labels = pad_sequence(sentences, batch_first=True, padding_value=torch.tensor(0))  # pad labels with 0 because pad token cannot be entities

        return padded_sentences, padded_labels

# # Convert word_embeddings and y_one_hot into a list of tuples TRAIN
print(len(word_embeddings_train[0]))
print(len(y_one_hot_train[0]))
word_embeddings_and_y_train = [(np.array(word_embedding).reshape(1,-1).tolist()[0], y_one_hot_train[i]) for i, word_embedding in enumerate(word_embeddings_train)]
print(type(word_embeddings_and_y_train))
# # Pass the list of tuples to EntityDataset
entity_dataset_train = EntityDataset(word_embeddings_and_y_train,word_vocab)
print(len(entity_dataset_train[0][0]))

# # Convert word_embeddings and y_one_hot into a list of tuples DEV
print(len(word_embeddings_dev[0]))
print(len(y_one_hot_dev[0]))

word_embeddings_and_y_dev = [(np.array(word_embedding).reshape(1,-1).tolist()[0], y_one_hot_dev[i]) for i, word_embedding in enumerate(word_embeddings_dev)]
print(type(word_embeddings_and_y_dev))
# # Pass the list of tuples to EntityDataset
entity_dataset_dev = EntityDataset(word_embeddings_and_y_dev,word_vocab)
print(len(entity_dataset_dev[0][0]))

# # Convert word_embeddings and y_one_hot into a list of tuples TEST
print(len(word_embeddings_test[0]))
print(len(y_one_hot_test[0]))

word_embeddings_and_y_test = [(np.array(word_embedding).reshape(1,-1).tolist()[0], y_one_hot_test[i]) for i, word_embedding in enumerate(word_embeddings_test)]
print(type(word_embeddings_and_y_test))
# # Pass the list of tuples to EntityDataset
entity_dataset_test = EntityDataset(word_embeddings_and_y_test,word_vocab)
print(len(entity_dataset_test[0][0]))


# class FFNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(FFNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # Apply ReLU activation function to the first fully connected layer
#         x = self.fc2(x)  # Apply the second fully connected layer without activation function
#         return x

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # First hidden layer with Sigmoid activation
        x = torch.sigmoid(self.fc2(x))  # Second hidden layer with Sigmoid activation
        x = torch.sigmoid(self.fc3(x))  # Third hidden layer with Sigmoid activation
        x = self.fc4(x)                  # Output layer without activation
        return x

# class FFNN(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
#         super(FFNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, hidden_size3)
#         self.fc4 = nn.Linear(hidden_size3, output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # First hidden layer with ReLU activation
#         x = F.relu(self.fc2(x))  # Second hidden layer with ReLU activation
#         x = F.relu(self.fc3(x))  # Third hidden layer with ReLU activation
#         x = self.fc4(x)          # Output layer without activation
#         return x

dataloader_train = DataLoader(entity_dataset_train, batch_size=200, shuffle=True)
# dataloader_train = DataLoader(entity_dataset_train, batch_size=100, shuffle=True)
# dataloader_train = DataLoader(entity_dataset_train, batch_size=50, shuffle=True)

#model = FFNN(input_size=(p + s + 1) * 100, hidden_size=256, output_size=len(pos_vocab))
model = FFNN(input_size=(p+s+1)*100, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=len(pos_vocab))

#model = FFNN(input_size=(p + s + 1) * 100, hidden_size=256, output_size=len(pos_vocab))
# print(output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
patience=5

best_accuracy = 0.0
no_improvement_count = 0
# dev_predictions=[]
test_predictions=[]
# real_dev_tags=[]
real_test_tags=[]
for epoch_num in range(50):
    model.train()
    for batch_num, (words, tags) in enumerate(dataloader_train):
        pred = model(torch.Tensor(words))
        loss = criterion(pred, tags)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch_num % 10 == 0:
        #    #print(f"{batch_num=}, {loss.item()=}")
    dev_predictions=[]
    real_dev_tags=[]

    model.eval()
    with torch.no_grad():
      total_correct = 0
      total_samples = 0
      for word, tag in entity_dataset_dev:
          pred = model(torch.Tensor(word))
          # print(pred)
          # print(pred.argmax().item())
          dev_predictions.append(reverse_pos_vocab[pred.argmax().item()])
          real_dev_tags.append(reverse_pos_vocab[tag.argmax().item()])
          total_samples += 1
          total_correct += (pred.argmax() == tag.argmax()).sum().item()
          dev_predictions.clear()
          real_dev_tags.clear()
      accuracy = total_correct / total_samples
      print(f"epoch: {epoch_num} Validation accuracy: {accuracy}")

      if accuracy > best_accuracy:
          best_accuracy = accuracy
          no_improvement_count = 0
          # Save the model
      else:
          no_improvement_count += 1

      if no_improvement_count >= patience:
          print(f"No improvement in validation accuracy for {patience} epochs. Stopping training.")
          break
      

test_predictions=[]
real_test_tags=[]
best_acc_test=0.0
patience=5
model.eval()
with torch.no_grad():
  total_correct = 0
  total_samples = 0
  for word, tag in entity_dataset_test:
      pred = model(torch.Tensor(word))
      # print(pred)
      # print(pred.argmax().item())
      test_predictions.append(reverse_pos_vocab[pred.argmax().item()])
      real_test_tags.append(reverse_pos_vocab[tag.argmax().item()])
      total_samples += 1
      total_correct += (pred.argmax() == tag.argmax()).sum().item()
      test_predictions.clear()
      real_test_tags.clear()
  accuracy = total_correct / total_samples
  print(f"Testing accuracy: {accuracy}")

  if accuracy > best_acc_test:
      best_acc_test = accuracy
      no_improvement_count = 0
      # Save the model
  else:
      no_improvement_count += 1

  if no_improvement_count >= patience:
      print(f"No improvement in Testing accuracy for {patience} epochs. Stopping training.")

torch.save(model, f"/content/my_model_hiddenlaye3_sigmoid.pt_{p}")


# Create the confusion matrix
conf_matrix = confusion_matrix(real_test_tags, test_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=pos_tags, yticklabels=pos_tags)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


####################################################################################################

##LSTM TAGGER

import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from conllu import parse
from gensim.models import Word2Vec
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define paths to your CoNLL-U files
train_path = "/content/drive/MyDrive/ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-train.conllu"
test_path = "/content/drive/MyDrive/ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-test.conllu"
dev_path = "/content/drive/MyDrive/ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-dev.conllu"


#Function to load and preprocess the CoNLL-U data
def load_conllu(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    sentences = []
    pos_tags = []
    for sentence in parse(data):
        words = [token["form"].lower() for token in sentence if isinstance(token, dict)]
        tags = [token["upos"] for token in sentence if isinstance(token, dict)]
        sentences.append(words)
        pos_tags.append(tags)
    #print(sentences[0],pos_tags[0])
    return sentences, pos_tags

train_sentences, train_pos_tags = load_conllu(train_path)
dev_sentences, dev_pos_tags = load_conllu(dev_path)
test_sentences, test_pos_tags = load_conllu(test_path)

# print(train_sentences)
# print(train_pos_tags)
train_data = list(zip(train_sentences, train_pos_tags))
dev_data = list(zip(dev_sentences, dev_pos_tags))
test_data = list(zip(test_sentences, test_pos_tags))
# Print the resulting list of tuples
#print(train_data)

# Create word index dictionary by iterating train_sentences
word_index = {}
for sentence in train_sentences:
    for word in sentence:
        if word not in word_index:
            word_index[word] = len(word_index)  # Assign unique index to each word

# Create POS index dictionary by iterating train_pos_tags
pos_index = {}
for tags in train_pos_tags:
    for tag in tags:
        if tag not in pos_index:
            pos_index[tag] = len(pos_index)   # Assign unique index to each POS tag

for sentence in dev_sentences:
    for word in sentence:
        if word not in word_index:
            word_index[word] = len(word_index)


for sentence in test_sentences:
    for word in sentence:
        if word not in word_index:
            word_index[word] = len(word_index)

# print( word_index)

# print(pos_index)


def prepare_sequence(seq, to_ix):
    """Input: takes in a list of words, and a dictionary containing the index of the words
    Output: a tensor containing the indexes of the word"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, activation='tanh'):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.target_size = target_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if activation == 'tanh':
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        elif activation == 'relu':
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, activation=nn.ReLU())
        elif activation == 'sigmoid':
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, activation=nn.Sigmoid())
        else:
            raise ValueError("Activation function not supported")

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    

# class LSTMTagger(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
#         super(LSTMTagger, self).__init__()

#         self.hidden_dim = hidden_dim

#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#         self.hidden2tag = nn.Linear(hidden_dim, target_size)

#     def forward(self, sentence):
#         embeds = self.word_embeddings(sentence)
#         lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
#         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_space, dim=1)
#         return tag_scores
    
EMBEDDING_DIM = 100
HIDDEN_DIM = 80

# Define the function to prepare sequence
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long).to(device)

# Initialize the model
#model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_index), len(pos_index)).to(device)
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_index), len(pos_index), activation='tanh')
#model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_index), len(pos_index), activation='sigmoid')
# lstm_sigmoid = LSTMTagger(embedding_dim, hidden_dim, vocab_size, target_size, activation='sigmoid')
# Define the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the training loop
for epoch in range(15):
    for sentence, tags in train_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_index)
        targets = prepare_sequence(tags, pos_index)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

    #print(f"{epoch=}, {loss.item()=}")

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
    for sentence, tags in dev_data:
        inputs = prepare_sequence(sentence, word_index)
        tag_scores = model(inputs)
        _, predicted = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(predicted)):
            for key, value in pos_index.items():
                if predicted[i] == value:
                    ret.append(key)
        total_samples += len(tags)
        # total_correct += (predicted[i] == tag).sum().item()
        for i in range(len(ret)):
          if ret[i]==tags[i]:
            total_correct += 1

        accuracy = total_correct / total_samples
    print(f"Epoch: {epoch=}, Validation accuracy: {accuracy}")


model.eval()
with torch.no_grad():
    correct = 0
    samples = 0

    for sentence, tags in test_data:
        inputs = prepare_sequence(sentence, word_index)
        tag_scores = model(inputs)
        _, predicted = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(predicted)):
            for key, value in pos_index.items():
                if predicted[i] == value:
                    ret.append(key)
        samples += len(tags)

        # Compare predicted tags with actual tags and count correct predictions
        for i in range(len(ret)):
            if ret[i] == tags[i]:
                correct += 1

    # Calculate accuracy
    accuracy = (correct / samples) * 100
    print("Accuracy on test set:", accuracy)


########
##RUN ON GOOGLE COLLAB 
## FFNN AND LSTM SEPARATE CODE

