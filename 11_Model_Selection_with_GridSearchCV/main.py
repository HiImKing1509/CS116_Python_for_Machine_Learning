import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, train_test_split

dataset=pd.read_csv('Social_Network_Ads.csv')
dataset=dataset.to_numpy()
print(dataset.shape)

param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear']}  
svc=svm.SVC()
X=dataset[:,[0,1]]
Y=dataset[:,-1]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

classification= GridSearchCV(svc, param_grid,refit = True, verbose = 3, n_jobs=-1)

print('Training:', x_train.shape, y_train.shape)
print('Testing: ', x_test.shape, y_test.shape)

classification.fit(x_train, y_train)
print(classification.best_params_)