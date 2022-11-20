import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report

dataset = load_wine()
df = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']],
                     columns= dataset['feature_names'] + ['customer_segment'])

X = dataset['data']
y = dataset['target']

# scale data
X_scale = StandardScaler().fit_transform(X)

# PCA
PCA = PCA(n_components=3)
X_pca = PCA.fit_transform(X_scale)

print(X.shape)
print(X_pca.shape)

def model_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=5)
    LR = LogisticRegression(random_state=0)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    print(f"accuracy = {accuracy_score(y_pred, y_test)}")
    
def k_fold(X, y):
    # KFold
    kf = KFold(n_splits=5, random_state=None)
    model = LogisticRegression(random_state=0) 
    acc_score = []

    for train_index , test_index in kf.split(X):
        X_train , X_test = X[train_index,:], X[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)
        
        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)
        
    avg_acc_score = sum(acc_score)/5
    print(acc_score)
    print(avg_acc_score)

model_train_test_split(X_pca, y)
k_fold(X_pca, y)