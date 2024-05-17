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


pd.set_option('display.max_colwidth', None)
data = pd.read_csv('/kaggle/input/train-csv/train.csv')

print("Top 20 rows:")
print(data.head(20))

class_index = data['Class Index']
description = data['Description']

unique_class_count = class_index.nunique()

print("Number of unique values in 'class_index':", unique_class_count)



df = pd.read_csv('/kaggle/input/train-csv/train.csv')

descriptions = df['Description'].iloc[0:15000]
print(descriptions.shape)
def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [re.sub(r'[^a-zA-Z0-9 ]', '', word) for word in tokens]
    tokens = [word for word in tokens if word]
    
    return tokens

def build_co_occurrence_matrix(descriptions, window_size=1):
    word_to_id = {}
    
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    word_freq = defaultdict(int)
    
    for desc in descriptions:
        tokens = preprocess(desc)
        for token in tokens:
            word_freq[token] += 1
        
    for desc in descriptions:
        tokens = preprocess(desc)
        for i, word in enumerate(tokens):
            if word_freq[word] < 5:
#                 word = '<UNK>'
                tokens[i]="<UNK>"
            if word not in word_to_id:
                word_id = len(word_to_id)
                word_to_id[word] = word_id

            start_index = max(0, i - window_size)
            end_index = min(len(tokens), i + window_size + 1)
            context_words = tokens[start_index:i] + tokens[i+1:end_index]
            
            for context_word in context_words:
                if word_freq[context_word] < 5:
                    context_word = '<UNK>'
                if context_word not in word_to_id:
                    word_id = len(word_to_id)
                    word_to_id[context_word] = word_id
                co_occurrence_matrix[word_to_id[word]][word_to_id[context_word]] += 1
    print(len(word_to_id))

    return co_occurrence_matrix, word_to_id

def co_occurrence_to_sparse_matrix(co_occurrence_matrix):
    rows = []
    cols = []
    data = []
    for i, row in co_occurrence_matrix.items():
        for j, value in row.items():
            rows.append(i)
            cols.append(j)
            data.append(value)
    return coo_matrix((data, (rows, cols)))

def apply_svd(matrix, k=150):
    svd = TruncatedSVD(n_components=k, random_state=42)
    word_vectors = svd.fit_transform(matrix)
    return word_vectors

def save_word_vectors(word_vectors, word_to_id, filename):
    with open(filename, 'w') as f:
        for word, word_id in word_to_id.items():
            vector_str = ' '.join(map(str, word_vectors[word_id]))
            f.write(f'{word} {vector_str}\n')

co_occurrence_matrix, word_to_id = build_co_occurrence_matrix(descriptions)

sparse_matrix = co_occurrence_to_sparse_matrix(co_occurrence_matrix)

word_vectors = apply_svd(sparse_matrix)


print(word_vectors[0])

model_save_name = 'svd-word-vectors.pt'
path = F"/kaggle/working/{model_save_name}"
torch.save({'vocab': word_to_id, 'embeddings': word_vectors}, path)

