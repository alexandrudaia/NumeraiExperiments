# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
train=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_training_data.csv')
tempTrain=train
print(train.head(3))
 
y=train['target']
train=train.drop(['target'],axis=1)
print(train.shape)
print(y.shape)
test=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_tournament_data.csv')
tempTest=test 

id=test['t_id']
test=test.drop(['t_id'],axis=1)
 
sub=pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/numerai_tournament_data.csv')
trainvec=np.array(train)
testvec=np.array(test)
y=np.array(y)
###################################fitting  tsne embeddings#######################################################
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from bhtsne import tsne
 

temp=tsne(np.concatenate((trainvec,testvec)),3,10)
newtr=np.hstack((trainvec,temp[0:136573,:]))
newts=np.hstack((testvec,temp[136573:,:]))


tempTrain['s1']=temp[0:136573,0]
tempTrain['s2']=temp[0:136573,1]
tempTrain['s3']=temp[0:136573,2]

tempTest['s1']=temp[136573:,0]
tempTest['s2']=temp[136573:,1]
tempTest['s3']=temp[136573:,2]

tempTrain.to_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/tempTrain.csv')

tempTest.to_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/tempTest.csv')

######################################## fitting  keras   neural network #############################################

 
from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
l.fit(newtr,y)
predLog=l.predict_proba(newts)


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn import cross_validation
def create_model():
    model=Sequential()
    model.add(Dense(15,input_dim=24,init='normal',activation='sigmoid'))
    model.add(Dense(10, init='normal', activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()

y_train=np_utils.to_categorical(tempTrain['target'])
model.fit(newtr,y_train,batch_size=250,nb_epoch=20,verbose=3)
pred=model.predict_proba(newts)
sub = pd.read_csv('/media/machine_learning/A80C461E0C45E7C01/all/numerai_datasets/example_predictions.csv')
sub['probability']=pred[:,1]
sub.to_csv("/home/machine_learning/Downloads/sub9keras.csv", index=False)
sub['probability']=predLog[:,1]
sub.to_csv("/home/machine_learning/Downloads/sub9Log.csv", index=False)


















