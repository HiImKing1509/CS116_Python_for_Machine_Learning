# ======================================================================================== Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Step 1: Load data
data = pd.read_csv("50_Startups.csv")
dataset = data.copy()
dataset_styles = data.copy()
dt = data.to_numpy()

X = dt[:, :-1]
y = dt[:, -1]

# Step 2: Data Visualization
sns.pairplot(dataset)
plt.savefig('pairlot')

# Step 3: Data preprocessing
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234)

# Step 5: Training model
model = LinearRegression()
hist = model.fit(X_train, y_train)
print('Model has been trained successfully')

# Step 6: Testing model
y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred) * 100)
print(mean_absolute_error(y_test, y_pred))