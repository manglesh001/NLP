1. Run ssfAPI.py by passing dataset folder as argument
    python3 ssfAPI.py Dataset
    Dataset: folder having all .pb files.
    link to dataset: https://drive.google.com/drive/folders/1unOhwoWIQ-8_LgUNjr3h-uhEKGHhqviA
    It will generate preprocessed.csv

2. Run perform_embeddings.py 
    python3 perform_embeddings.py
    It will read preprocessed.csv and create embeddings as embeddings.csv
    link to embeddings.csv: https://drive.google.com/drive/folders/1EF4kH_HsuNoQD3YJ4TeyybgikPjDUiD9

3. Run all b*.py files for identifaction and c*.py for classification
    python3 b*.py
    python3 c*.py
    It will read embeddings.py and create models.pkl and results.txt
    link to model folder: https://drive.google.com/drive/folders/1UMIP9NCsbfREUo2iXVcrIrxo8Cepz8Iq
    
4. Link for ppt: https://drive.google.com/drive/folders/1UMIP9NCsbfREUo2iXVcrIrxo8Cepz8Iq
