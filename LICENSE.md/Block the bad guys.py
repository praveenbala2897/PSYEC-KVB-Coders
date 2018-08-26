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
