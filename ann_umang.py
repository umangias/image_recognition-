import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:, 3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
x[:,1]=label.fit_transform(x[:,1])
label2=LabelEncoder()
x[:,2]=label2.fit_transform(x[:,2])
#to change object into the float format array
one=OneHotEncoder(categorical_features=[1])
x=one.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#building ann
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
#initialising the network
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))


#classifier compilation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 classifier.fit(x_train, y_train,batch_size = 10, nb_epoch = 100 )

#predicting
 y_pred=classifier.predict(x_test)
 y_pred=(y_pred>0.5)
 
 from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

