import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import string
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F

class Preprocess:
    def __init__(self,train_file_path,test_file_path,padding=False):
        self.train_file_path=train_file_path
        self.test_file_path=test_file_path
        self.padding=padding
        self.word2idx={"<PAD>":0,"<UNK>":1,"<SOS>":2,"EOS":3}
    
    def process_data(self,split='train',min_cnt=3):
        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)
        ps = PorterStemmer()
        
        if split=='train':
            file_path = self.train_file_path
        else:
            file_path = self.test_file_path
        data = pd.read_csv(file_path)
        reviews = data['Description']
        print("len of reviews ",len(reviews))
        labels = data['Class Index']
        word_count={}
        cleaned_sentences=[]
        cnt=0;
        for idx, desc in enumerate(reviews):
            curr_record=[]
            curr_record.append("<SOS>")
            for sentence in sent_tokenize(desc):
                sentence = re.sub(r'[\\-]', ' ', sentence)
                sentence = re.sub(r'[^a-zA-Z0-9 ]', '', sentence)
                sentence = word_tokenize(sentence)
                sentence = [word for word in sentence if word not in stop_words and word not in punctuations]
                for word in sentence:
                    if word not in word_count:
                        word_count[word]=1
                    else:
                        word_count[word]+=1
                curr_record.extend(sentence)
            curr_record.append("<EOS>")
            cleaned_sentences.append(curr_record)
            cnt+=1
            if(cnt==20000):
                break
        if split=='train':
            for word,cnt in word_count.items():
                if(cnt>=min_cnt):
                    self.word2idx[word]=len(self.word2idx)
        for idx,sentence in enumerate(cleaned_sentences):
            cleaned_sentences[idx]=[word if word in self.word2idx else "<UNK>" for word in sentence]
        if split=='train':
            return cleaned_sentences,self.word2idx
        else:
            return cleaned_sentences

class ElmoDataset(Dataset):
    def __init__(self, X, Y, word2idx, pad=True, max_len=50):
        self.X = X
        self.Y = Y
        self.word2idx = word2idx
        self.pad = pad
        self.max_len = max_len

        # Convert words to indices
        self.X_indexed = []
        for sentence in self.X:
            indexed_sentence = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence]
            self.X_indexed.append(torch.tensor(indexed_sentence))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_forward = self.X_indexed[idx]
        X_backward = X_forward.flip(0)
        if len(X_forward) > self.max_len:
            X_forward = X_forward[:self.max_len]
            X_backward = X_backward[:self.max_len]
        
        # Pad or concatenate sequences
        if self.pad:
            pad_length_forward = max(0, self.max_len - len(X_forward))
            pad_length_backward = max(0, self.max_len - len(X_backward))
            forward = F.pad(X_forward, (0, pad_length_forward))
            backward = F.pad(X_backward, (0, pad_length_backward))
        else:
            forward = X_forward[:self.max_len]
            backward = X_backward[:self.max_len]

        Y = self.Y[idx]
        Y_one_hot = torch.zeros(4)
        Y_one_hot[Y-1] = 1
        return forward, backward, Y_one_hot