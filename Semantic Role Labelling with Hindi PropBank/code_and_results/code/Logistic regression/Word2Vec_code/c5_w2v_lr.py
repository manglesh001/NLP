import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

filepath = "emb_w2v.csv"
dataset = pd.read_csv(filepath)
dataset = dataset.dropna()
dropped_fields = ['label', 'chunk', 'srl', 'postposition', 'head-POS', 'predicate']
temp = ['dependency-head', 'dependency']

for j in range(1):
    dropped = temp    
    vectors = dataset.drop(dropped_fields, axis=1)
    target = dataset['srl']
    for col in dropped:
        vals = vectors[col].values
        vectors[col] = LabelEncoder().fit_transform(vals)
    X_train, X_test, y_train, y_test = train_test_split(vectors, target, test_size=0.30, random_state=42)
    logistic_classifier = LogisticRegression(max_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    y_pred = logistic_classifier.predict(X_test)

    joblib.dump(logistic_classifier, f'classification/model_w2v_logistic/model_5_{dropped}.pkl')    
    with open(f'classification/results_w2v_logistic/{dropped}_results.txt', 'a') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + '\n')
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + '\n')
