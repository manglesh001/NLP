import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


filepath = "emb.csv"
dataset = pd.read_csv(filepath)
print(dataset.columns)
dropped_fields=['label','chunk','srl','postposition','head-POS','dependency','dependency-head','predicate']
vectors = dataset.drop(dropped_fields, axis=1)
target = dataset['srl']
targ = target.values
target = LabelEncoder().fit_transform(targ)
X_train, X_test, y_train, y_test = train_test_split(vectors, target, test_size=0.30)
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)
y_pred = logistic_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(logistic_classifier, f'classification/model_fasttext_logistic/model_2_head.pkl')
with open(f'classification/results_fasttext_logistic/head_results.txt', 'a') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + '\n')
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred) + '\n')
