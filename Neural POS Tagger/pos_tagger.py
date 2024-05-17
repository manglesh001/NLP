import torch
from gensim.models import KeyedVectors
import json
import sys

class FeedForwardNN1(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_tags):
        super(FeedForwardNN1, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(hidden_size, num_tags)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        return x
    
class MyLSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, num_layers):
        super(MyLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.hidden2tag = torch.nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
    
def rnn():
    file_path = 'lstm_idx.txt'

    with open(file_path, 'r') as file:
        content = file.read()
    data = json.loads(content)

    word_to_idx = data['word_to_idx']
    tag_to_idx = data['tag_to_idx']

    lstm_model_path = "lstm_model_3.pt"
    model = torch.load(lstm_model_path)
    while(True):
        user_input=input("Enter a sentence or enter exit to exit\n")
        if(user_input.lower()=="exit"):
            exit(1)
        sent = user_input.split()
        word_idx=[]
        for word in sent:
            if(word in word_to_idx):
                word_idx.append(word_to_idx[word])
            else:
                word_idx.append(word_to_idx["<UNK>"])
        predicted=[]
        tag_scores = model(torch.tensor(word_idx))
        _, indices = torch.max(tag_scores, 1)
        for i in range(len(indices)):
            for key, value in tag_to_idx.items():
                if indices[i] == value:
                    predicted.append(key)
        for i in range(len(predicted)):
            print(sent[i],predicted[i])

def ffn():
    model_path = "ffn_model_1.pt"
    model = torch.load(model_path)
    
    word_vectors = KeyedVectors.load_word2vec_format('word2vec_embeddings.txt', binary=False)
    
    p = 3
    reverse_y_vocab = {0: 'PRON', 1: 'VERB', 2: 'DET', 3: 'NUM', 4: 'ADJ', 5: '<STAG>', 6: 'INTJ', 7: '</STAG>', 8: 'ADV', 9: 'PART', 10: 'PROPN', 11: 'NOUN', 12: 'ADP', 13: 'CCONJ', 14: 'AUX'}
    while(True):
        user_input=input("Enter a sentence or enter exit to exit\n")
        if(user_input.lower()=="exit"):
            exit(1)
        pred = []
        sent=user_input.split()
        for i in range(p):
            sent.insert(0,"<S>")
            sent.append("</S>")
        for i in range(p, len(sent) - p):
            curr = sent[i - p:i + p + 1]  
            input_vectors = []  
            for word in curr:
                if word in word_vectors:
                    input_vectors.append(word_vectors[word])
                else:
                    input_vectors.append(word_vectors["<UNK>"])
            input_tensor = torch.tensor(input_vectors)
            input_tensor = input_tensor.view(-1, input_tensor.size(0) * input_tensor.size(1))
            output = model(input_tensor)
            predicted_tag_idx = output.argmax().item()
            predicted_tag = reverse_y_vocab[predicted_tag_idx]
            pred.append(predicted_tag)
        for i in range(len(pred)):
                print(sent[p+i],pred[i])
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Incorrect arguments")
        exit(1)

    elif sys.argv[1]=="-f":
        ffn()

    elif sys.argv[1]=="-r":
        rnn()

        

