import pandas as pd
from gensim.models import Word2Vec
import numpy as np

df = pd.read_csv('preprocessed.csv')
unique_words = set(df['word'])
model = Word2Vec(sentences=[unique_words], vector_size=300, window=10, min_count=1, workers=4)
word_embeddings = {word: model.wv[word] for word in unique_words}

def get_embedding(word):
    return word_embeddings[word] if word in word_embeddings else np.zeros(300)  

df['embedding'] = df['word'].apply(get_embedding)
embedding_cols = [f'emb{i}' for i in range(300)]
embedding_df = pd.DataFrame(df['embedding'].tolist(), columns=embedding_cols)

def is_predicate_arg(row):
    if row['is_arg'] == 1:
        return 1
    else:
        return 0

df['label'] = df.apply(is_predicate_arg, axis=1)
df = pd.concat([embedding_df, df.drop(columns=['word', 'embedding', 'is_arg'])], axis=1)
columns_order = embedding_cols + ['label'] + [col for col in df.columns if col not in embedding_cols + ['label']]
df = df.reindex(columns=columns_order)

df.to_csv('embeddings_word2vec.csv', index=False)
