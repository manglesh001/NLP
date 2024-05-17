## README

1. Training and testing the model for both the networks was done on google colab.
2. preprocess.py contains the class Preprocess with the required functions to tokenize the sentences, remove stop words, special characters and punctuations. It also contains the ELMODataset class for the dataset initialization to train both the ELMO Model and the classfication model.
3. ElMO.py contains the ELMO model class, the training and testing code for ELMO model.
4. classification.py contains the classification model which uses a bidirectional LSTM.

### ELMO Model Architecture
-  __init__: Constructor method where you define the architecture of the model.
    - vocab_size: The size of the vocabulary.
    - embedding_dim: Dimensionality of the word embeddings.
    - batch_size: Size of the input batch.
    - embedding_matrix: Pre-trained word embeddings (embedding matrix) to initialize the embedding layer.
    - self.embedding: Embedding layer initialized with pre-trained word embeddings. The freeze=True argument freezes the embedding layer during training.
    - self.lstm1 and self.lstm2: Two LSTM layers. These are bidirectional LSTM layers (bidirectional=False), meaning they process input sequences from left to right only.
    - self.linear1 and self.linear_out: Linear layers used for dimensionality reduction and output prediction, respectively.
 - forward: Forward method defining the forward pass of the model.
    - X: Input tensor representing a batch of sequences of word indices.
    - self.embedding(X): Applies the embedding layer to the input tensor.
    - self.lstm1(X) and self.lstm2(X): Applies the LSTM layers to the embedded sequences.
    - self.linear1(X): Applies a linear transformation to the output of the LSTM layers.
    - self.linear_out(X): Applies another linear transformation to obtain the final output logits.

### ELMO Training
- Two separate LSTM models with two layers, optimizers and loss crierions were created for forward and backward training.
- The forward model was trained with the sentences in forward direction and backward model trained in backward direction.

### Classificaion
- The backward and forward models embedding layers and the two lstm layers were passed to the classifier during initialization for calculating the ELMO embeddings.
- The classifier uses a bidirection lstm for classification task.

### Steps to run the file
    ```python3 ELMO.py```
    ```python3 classification.py```