import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

df = pd.read_csv("adult.csv")

df_features = ['age', 'fnlwgt','education-num','capital-gain', 'capital-loss', 'hours-per-week']

X = df[df_features]

Y = df.salary

# Set Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X , Y, 
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=1)

# Show the Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)

from sklearn import tree

# Building Decision Tree model 
dtc = tree.DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

pickle.dump(dtc, open('model.pkl','wb'))