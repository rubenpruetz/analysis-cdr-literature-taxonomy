#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Linear SVM text classifier used to predict the relevance of the unseen literature"""

# import libraries
import string
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')

# import data on already labeled literature
filepath = ''  # specify filepath of csv file
filename = 'initial_training_sample_n1010.csv'
df = pd.read_csv(filepath + filename)
df = df[['relevance',  # names of columns might differ, the order is important
         'title',
         'abstract',
         'keywords']]



##### preprocess and clean titles, abstracts, and keywords #####

df['text'] = df[df.columns[1:]].apply(  # join columns of titles, abstracts, and keywords
    lambda x: ' '.join(x.dropna().astype(str)), axis=1)
raw_list = df['text'].tolist()  # add created text column to list

for i in range(len(raw_list)):  # lower characters
    raw_list[i] = raw_list[i].lower()

punct_list = []  # remove punctuation
for text in raw_list:
    for p in text:
        if p in string.punctuation:
            text = text.replace(p, '')
    punct_list.append(text)

def clean(text):  # filter stopwords and stem words
    ps = PorterStemmer()
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    stemmed_text = [ps.stem(w) for w in filtered_text]
    return ' '.join(stemmed_text)
text_list = [clean(text) for text in punct_list]

relevance_list = df['relevance'].tolist()  # add relevance column to list



##### build and train svm classifier #####

# split the text and relevance information in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_list,
                                                    relevance_list,
                                                    test_size=0.1,
                                                    random_state=0)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge')),])  # define classifier model

param = {  # specify parameter options for grid search to optimize classifier performance
        'vect__max_df': (0.1, 0.5, 1.0),
        'vect__stop_words': [None, stop_words],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False), }

text_clf = GridSearchCV(text_clf, param, n_jobs=-1)

text_clf = text_clf.fit(X_train, y_train)  # train classifier with training data
predicted = text_clf.predict(X_test)  # predicting relevance of test set with classifier
print('best param : ',text_clf.best_params_)  # evaluate best performing parameters



##### evaluate and save classifier #####

# visualize performance of classifier model
fig1, ax1 = plt.subplots(figsize=(9, 3), dpi=300)
plot_confusion_matrix(text_clf, X_test, y_test, ax=ax1)
plt.show()

report = classification_report(y_test, predicted, target_names=np.unique(y_test), output_dict=True)
plt.figure(figsize=(8, 5), dpi=300)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)

# save trained classifier for further use in predicting the relevance of unseen literature
filename = ''  # specify output filename with .sav
pickle.dump(text_clf, open(filename, 'wb'))
