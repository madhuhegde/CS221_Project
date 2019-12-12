import os
#import numpy as np 
import pandas as pd 
import math
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./data/train.csv", verbose = True)
train_df, val_df = train_test_split(train_df, test_size = 0.1, random_state = 42)

X_train = train_df['question_text']
y_train = train_df['target']
X_test = val_df['question_text']
y_test = val_df['target']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

vect = CountVectorizer().fit(X_train)
X_train_vec = vect.transform(X_train)
print("Number of features used: ", len(vect.get_feature_names()))

clfrNB = MultinomialNB(alpha = 0.1)
clfrNB.fit(X_train_vec, y_train)
preds = clfrNB.predict(vect.transform(X_test))
score = f1_score(y_test, preds)
print("F1 Score: ", score)
