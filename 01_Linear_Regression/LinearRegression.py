import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('./Salary_Data.csv')

# X.reshape(-1, 1) for [[]]
X = np.array(data.iloc[:,0].values).reshape(-1,1)
Y = np.array(data.iloc[:,1].values)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
plt.scatter(X_train, Y_train, color = "red", s = 4)
plt.title('Dataset')
plt.subplot(1, 1, 1)
plt.show()

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(Y_pred)
plt.scatter(X_test, Y_test, color = 'red', s = 4)
plt.plot(X_test, Y_pred, color = 'blue')
plt.scatter(X_test, Y_pred, color = 'black', s = 4)
plt.title('Predict test data')
plt.subplot(1, 1, 1)
plt.show()

def Score(reg, X_, Y_):
    rsq = reg.score(X_, Y_)
    return 1 - (1 - rsq) * ((X_.shape[0] - 1) / (X_.shape[0] - X.shape[1] - 1))

print(Score(model, X_train, Y_train))
print(Score(model, X_test, Y_test))
print(model.predict(np.array([8]).reshape(-1, 1)))