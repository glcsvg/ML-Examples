# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:18:34 2018

@author: Pc
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv(r'C:\Users\Pc\Desktop\pythonudemy1\Churn_Modelling.csv')

X=data.iloc[:,3:13].values
Y=data.iloc[:,-1:].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X =X[:,1:]


from  sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(6,init= 'uniform',activation='relu',input_dim=11))

classifier.add(Dense(6,init= 'uniform',activation='relu'))
classifier.add(Dense(1,init= 'uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


classifier.fit(X_train,y_train,epochs=80)

y_ped=classifier.predict(X_test)

y_ped=(y_ped>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_ped)
print(cm)









