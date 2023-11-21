from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# assing data set to the variables
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# this reads only the top 5 elements of the dataset
data_fake.head()
data_true.head()

data_fake["class"] = 0
data_true['class'] = 1

# it gives the shape i.e. number of rows and columns of the dataset
data_fake.shape, data_true.shape

# for manual testing purpose
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)


data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)


data_fake.shape, data_true.shape

data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

data_fake_manual_testing.head(10)
data_true_manual_testing.head(10)

# merging both datasets
data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge.head(10)

data_merge.columns

# removing unwanted columns from dataset (for testing purpose)
data = data_merge.drop(['title', 'subject', 'date',], axis=1)

data.isnull().sum()

data = data.sample(frac=1)  # random shuffeling of data

data.head()

data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

data.columns

data.head

# function to process the text


def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
# this will remove these types of special characters from the dataset


data['text'] = data['text'].apply(wordopt)

# defining independent and dependent variables
x = data['text']
y = data['class']

# splitting, training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# text to vectors

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# model Logistic Regression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

# for accuracy score
LR.score(xv_test, y_test)

print('accuracy score for logistic regression :', LR.score(xv_test, y_test))

print(classification_report(y_test, pred_lr))

# same for decission tree, gradient boosting classifier, random forest classifier
# decission tree

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

DT.score(xv_test, y_test)

print('accuracy score for DecisionTreeClassifier :', DT.score(xv_test, y_test))

print(classification_report(y_test, pred_dt))

# gradient boosting classifier

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

pred_gb = GB.predict(xv_test)

GB.score(xv_test, y_test)

print('accuracy score for GradientBoostingClassifier :', GB.score(xv_test, y_test))

print(classification_report(y_test, pred_gb))

# random forest classifier

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

pred_rf = RF.predict(xv_test)

RF.score(xv_test, y_test)

print('accuracy score for RandomForestClassifier :', RF.score(xv_test, y_test))

print(classification_report(y_test, pred_rf))

# all models are working message
print('All Models are working')


# output lable
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


# manual testing
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GB[0]),
        output_label(pred_RF[0])
    ))


# taking input
news = str(input('Enter the news you want to check :'))
manual_testing(news)

print('code is running completely')