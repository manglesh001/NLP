import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys

filepath = "embeddings_word2vec.csv"
dataset = pd.read_csv(filepath)
params_to_test = ['dependency-head','dependency']

for z in range(1):
    dropped=params_to_test    
    dropped_fields = ['label','chunk','srl','postposition','head-POS','predicate']
    X = dataset.drop(dropped_fields, axis=1)
    y = dataset['srl']
    vals=X[dropped[0]].values
    X[dropped[0]]=LabelEncoder().fit_transform(vals)
    vals2=X[dropped[1]].values
    X[dropped[1]]=LabelEncoder().fit_transform(vals2)
    y = LabelEncoder().fit_transform(y.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    joblib.dump(svclassifier, f'Classification/model_word2vec_svc/model_5_head_both_dependencies.pkl')
    with open(f'Classification/results_word2vec_svc/head_both_dependencies_results.txt', 'a') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + '\n')
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + '\n')
