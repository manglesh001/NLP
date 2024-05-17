import pandas as pd
import fasttext.util

# Load pre-trained FastText model for Hindi
# Make sure you have downloaded the FastText model for Hindi beforehand
# fasttext.util.download_model('hi', if_exists='ignore')  # Download if not already downloaded
ft = fasttext.load_model('cc.hi.300.bin')  # Load pre-trained FastText model for Hindi

# Load the CSV file
df = pd.read_csv('preprocessed.csv')

# Function to get FastText embeddings for a word
def get_embedding(word):
    return ft.get_word_vector(word)

# Apply the function to each word in the 'word' column and create a new column for embeddings
for i in range(300):
    col_name = f'emb{i}'
    df[col_name] = df['word'].apply(get_embedding).apply(lambda x: x[i])

# Remove the 'word' column

# Create 'label' column based on 'is_arg'
df['label'] = df['is_arg']

df.drop(columns=['word', 'is_arg'], inplace=True)
# Reorder the columns
cols = ['emb' + str(i) for i in range(300)] + ['label'] + list(df.columns.difference(['emb' + str(i) for i in range(300)] + ['label']))

# Save the DataFrame with embeddings to a new CSV file
df[cols].to_csv('emb_fasttext.csv', index=False)