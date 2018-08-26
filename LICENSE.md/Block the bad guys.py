import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib notebook

np.random.seed(19)

train = pd.read_csv('C:/Users/B.BALAPRAKASH/Desktop/train.csv')
test = pd.read_csv('C:/Users/B.BALAPRAKASH/Desktop/test.csv')
print("Training samples = ", train.shape[0])
print("Test samples = ", test.shape[0])

train_rows =train.shape[0]
labels = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] 
train.drop(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
test_id = test.pop('id')
train.head()


print("Null values in training data", train.isnull().sum(), sep="\n")
print("Null values in testing data", test.isnull().sum(), sep="\n")

data = pd.concat([train, test])
del train
del test
data.shape

import re
import nltk

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_input(comment):
# remove the extra spaces at the end.
    comment = comment.strip()
# lowercase to avoid difference between 'hate', 'HaTe'
    comment = comment.lower()
# remove the escape sequences. 
    comment = re.sub('[\s0-9]',' ', comment)
# Use nltk's word tokenizer to split the sentence into words. It is better than the 'split' method.
    words = nltk.word_tokenize(comment)
# removing the commonly used words.
    #words = [word for word in words if not word in stop_words and len(word) > 2]
    words = [word for word in words if len(word) > 2]
    comment = ' '.join(words)
    return comment
  
print("SAMPLE PREPROCESSING")
print("\nOriginal comment: ", data.comment_text.iloc[0], sep='\n')
print("\nProcessed comment: ", preprocess_input(data.comment_text.iloc[0]), sep='\n')

data.comment_text = data.comment_text.apply(lambda row: preprocess_input(row))


data.head()

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=0.1, max_df=0.7, 
                       analyzer='char',
                       ngram_range=(1, 3),
                       strip_accents='unicode',
                       sublinear_tf=True,
                       max_features=30000
                      )


test = data[train_rows:]
train = data[:train_rows]
del data

vect = vect.fit(train.comment_text)
train = vect.transform(train.comment_text)
test = vect.transform(test.comment_text)

print('Training feature set = ', train.shape)
print('Testing feature set = ', test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_pred = pd.read_csv('C:/Users/B.BALAPRAKASH/Desktop/sample_submission.csv')

for c in cols:
    clf = LogisticRegression(C=4, solver='sag')
    clf.fit(train, labels[c])
    y_pred[c] = clf.predict_proba(test)[:,1]
    score = np.mean(cross_val_score(clf, train, labels[c], scoring='roc_auc', cv=5))
    print("ROC_AUC score for", c, "=",  score)
    
    y_pred.head()
    
    y_pred.to_csv('my_submission.csv', index=False)


