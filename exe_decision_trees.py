# Basic data processing and visuliazation
import numpy as np 
import pandas as pd 
import random 
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns 
pd.options.display.max_columns = None

# Feature engineering and auto processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from gensim.models import Word2Vec

# ML model building
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ML algorithms
from sklearn.tree import DecisionTreeClassifier


def import_data(path, review_data=True, **csv_args):
    """load dataset"""
    data = pd.read_csv(path, **csv_args)
    
    if review_data:
        print("data shape", data.shape)
        print("*"*10)
        print("data size in MB: \n", 
            round(data.memory_usage(deep=True).sum()/(1024*1024), 2) )
        print("*"*10)
        df_summary = pd.DataFrame(
            {
            'dtypes': data.dtypes, 
            'count_na': data.isna().sum(axis='index'),
            'count_zero': (data==0).sum(axis='index')
            }
            )
        print(df_summary)
    return data

def clean_text(df):
    df = df.astype('object')
    df = df.fillna(' ')
    def cleaning(series):
        # remove white space and lower
        series = series.apply(lambda x: 
            ' '.join(x.lower().split()))
        # remove numers
        num_pattern = r"(?:\$?\d+\.?\,?\d*%?)"
        series = series.apply(lambda x: re.sub(num_pattern, '', x))
        # remove punctuations
        def remove_punc(text):
            translator = str.maketrans('', '', string.punctuation)
            return text.translate(translator)
        series = series.apply(remove_punc)
        return series
    return df.apply(cleaning)


def clean_data(df, remove_cols, cat_cols):
    df.drop(remove_cols, axis='columns',inplace=True)
    df[cat_cols] = df[cat_cols].astype('category')
    return df

def train_tree(X_train, y_train, **tree_args):
    """train tree model"""
    clf = DecisionTreeClassifier(
        random_state=123, **tree_args
    )
    clf.fit(X_train, y_train)
    return clf


# Load data for train and test    
path = './datasets/titanic/titanic_dataset.csv'
data = import_data(path, review_data=True)
data['combined_text'] = (
    data[['Name', 'Sex', 'Cabin', 'Embarked']]
    .fillna(' ')
    .apply(lambda row: 
        ' '.join(row), axis=1))

# define preprocessing columns
text_countvector_cols = ['Cabin']
text_tfidf_cols = ['Name']
text_word2vect_cols = ['combined_text']
text_cols = text_countvector_cols + text_tfidf_cols + text_word2vect_cols
cat_cols = ['Pclass', 'Sex', 'Embarked']
target_col = ['Survived']

# split train and text
X = data.drop(target_col, axis='columns')
Y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size = 0.2, random_state=123)

# model preprocessing pipeline using Pipeline
# ColumnTransformer -> Pipeline
# ColumnsTransformer
text_clean_transformer = ColumnTransformer(transformers=[
    ('clean_text', FunctionTransformer(clean_text), text_cols)
])
text_count_transformer = ColumnTransformer(transformers=[
    'text_countvector', CountVectorizer(), text_countvector_cols
])
text_tfidf_transformer = ColumnTransformer(transformers=[
    ('tfidf', TfidfVectorizer(ngram_range=(1,2)), text_tfidf_cols)
])


