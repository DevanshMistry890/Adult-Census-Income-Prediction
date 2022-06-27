# importing the dataset
import pandas
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import pickle
import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.debug(' Model.py File execution started ')


df = pandas.read_csv('adult.csv')
logging.debug(' Database Loaded ')

df = df.drop(['fnlwgt', 'education-num'], axis=1)

col_names = df.columns

for c in col_names:
	df = df.replace("?", numpy.NaN)
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

df.replace(['Divorced', 'Married-AF-spouse',
			'Married-civ-spouse', 'Married-spouse-absent',
			'Never-married', 'Separated', 'Widowed'],
		['divorced', 'married', 'married', 'married',
			'not married', 'not married', 'not married'], inplace=True)

category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
				'relationship', 'sex', 'country', 'salary']
labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
	df[col] = labelEncoder.fit_transform(df[col])

	le_name_mapping = dict(zip(labelEncoder.classes_,
							labelEncoder.transform(labelEncoder.classes_)))

	mapping_dict[col] = le_name_mapping
print(mapping_dict)

df.columns

logging.debug(' Database Pre-Processing Done ')

X = df.values[:, 0:12]
Y = df.values[:, 12]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)



# Show the Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)

from sklearn.ensemble import ExtraTreesClassifier
# Building the model
extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy', max_features = 2)
extra_tree_forest.fit(X, Y)
predictions_e = extra_tree_forest.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, predictions_e))

pickle.dump(extra_tree_forest, open('model.pkl','wb'))
logging.debug(' Execution of Model.py is finished ')