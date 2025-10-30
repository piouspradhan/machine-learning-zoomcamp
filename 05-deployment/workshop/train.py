#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
import sklearn

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


def load_data():
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    return df




def train_model():

    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]



    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )




    # dv = DictVectorizer()

    # train_dict = df[categorical + numerical].to_dict(orient='records')
    # X_train = dv.fit_transform(train_dict)

    # model = LogisticRegression(solver='liblinear')
    # model.fit(X_train, y_train)


    train_dict = df[categorical + numerical].to_dict(orient='records')
    y_train = df.churn.values
    pipeline.fit(train_dict, y_train)
    return pipeline


def save_model(model, pipeline):

    # with open ('model.bin', 'wb') as f_out:
    #     pickle.dump((dv, model), f_out)

    with open (model, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

    # get_ipython().system('ls -lh')

# dv
# model
#pipeline

df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)
