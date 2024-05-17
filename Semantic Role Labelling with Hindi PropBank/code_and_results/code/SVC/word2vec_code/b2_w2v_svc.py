import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys

filepath = "embeddings_word2vec.csv"
dataset = pd.read_csv(filepath)    
dropped_fields = ['label', 'chunk', 'srl', 'postposition', 'head-POS', 'dependency', 'dependency-head', 'predicate']
X = dataset.drop(dropped_fields, axis=1)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
    
joblib.dump(svclassifier, f'identification/model_word2vec_svc/model_2_head.pkl')
with open(f'identification/results_word2vec_svc/head_results.txt', 'a') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + '\n')
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred) + '\n')
