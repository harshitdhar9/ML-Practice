# -*- coding: utf-8 -*-
"""CarPricePrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nhElIsfCpJe7_kXwY11i9yKEeG8jz5Z2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/content/quikr_car.csv')
df.head()

df.shape
df1=df.copy()

df['year']=df['year'].astype(int)
df=df[df['Price']!='Ask For Price']
df['kms_driven']=df['kms_driven'].str.split().str.get(0).str.replace(',','')
df=df[df['kms_driven'].str.isnumeric()]
df['kms_driven']=df['kms_driven'].astype(int)
df=df[~df['fuel_type'].isna()]
df['name']=df['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
df=df.reset_index(drop=True)

df.head()

X=df[['name','company','year','kms_driven','fuel_type']]
y=df['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])

column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')

lr=LinearRegression()

pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

r2_score(y_test,y_pred)

scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))

np.argmax(scores)