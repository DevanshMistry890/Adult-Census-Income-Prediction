# This file is showing how model process data form raw to training,
# for modelling there is saperate file
# importing Required libraries
import logging
import numpy
import pandas
import imblearn
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# log file initialization 
logging.basicConfig(filename='test.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.debug(' Model.py File execution started ')

# loading database with pandas library
df = pandas.read_csv('adult.csv')
logging.debug(' Database Loaded ')

# dropping unnecessary and redundant  columns
df = df.drop(['fnlwgt', 'education-num'], axis=1)

# handling missing values in df
col_names = df.columns
for c in col_names:
	df = df.replace("?", numpy.NaN)
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))


# Discretization
df.replace(['Divorced', 'Married-AF-spouse',
			'Married-civ-spouse', 'Married-spouse-absent',
			'Never-married', 'Separated', 'Widowed'],
		['divorced', 'married', 'married', 'married',
			'not married', 'not married', 'not married'], inplace=True)

category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
				'relationship', 'sex', 'country', 'salary']

# labelEncoding
labelEncoder = preprocessing.LabelEncoder()

# mapping 
"""
	mapping categorical variables to numerical value under labelencoder
	return dictionary   
""" 
mapping_dict = {}
for col in category_col:
	df[col] = labelEncoder.fit_transform(df[col])

	le_name_mapping = dict(zip(labelEncoder.classes_,
							labelEncoder.transform(labelEncoder.classes_)))

	mapping_dict[col] = le_name_mapping

logging.debug(' Database Pre-Processing Done ')

# model featuring
X = df.values[:, 0:12]
Y = df.values[:, 12]

# Data Spliting For model training
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

# model fitting using LGBMClassifier
clf = lgb.LGBMClassifier()
clf.fit(X_train_SMOTE, y_train_SMOTE)


# Show the Training and Testing Data
print('Shape of training feature:', X_train_SMOTE.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train_SMOTE.shape)
print('Shape of training label:', y_test.shape)

# Printing Accuracy
predictions_e = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions_e))

# pkl export & finish log
pickle.dump(clf, open('model.pkl','wb'))
logging.debug(' Execution of Model.py is finished ')