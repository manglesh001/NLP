## README

### Tokenizer.py
1. This file has the code for tokenizing the given input sentences into list which contains the list of tokens for every sentence.
2. On running the file an input prompt will be shown where the user can enter a sentence and the tokenized output will be printed.


```python3 Tokenizer.py```

### language_model.py
1. This file contains the code for creating the language model and estimating the probability of tan input sentence.
2. To train the model run the below command. This will create two files with the model paramters, one for good_turing probabilities and one for linear interpolation weights and probabilities.


```python3 language_model.py tr <corpus_path>```

3. To load the model from the file, run the below command and it will load the required model parameters based on the lm_type parameter and the user will be prompted to give an input sentence whose probability will be calculated.


```python3 language_model.py <lm_type> <corpus_path>```

4. The txt files for perplexities are generated during the training process. We have to chaging the file names, probability and test/train variables in the method calculate_perplexities() for generating the perplexity files.

### generator.py
1. This file contains the code for generating the possible next words given a sentence. 
2. This will print a k length dictionary of possible next words along with their probability. lm_type can be 
    1. none for no smoothing
    2. g for good turing
    3. i for linear interpolation


```python3 generator.py <lm_type> <corpus_path> k```
