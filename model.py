# importing the dataset
import pandas
import numpy
from sklearn import preprocessing
import pickle

df = pandas.read_csv('adult.csv')
df.head()

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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = df.values[:, 0:12]
Y = df.values[:, 12]

X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size = 0.3, random_state = 100)

dt_clf_gini = DecisionTreeClassifier(criterion = "gini",
									random_state = 100,
									max_depth = 5,
									min_samples_leaf = 5)

dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

print ("Decision Tree Accuracy is ",
			accuracy_score(y_test, y_pred_gini)*100 )

from sklearn.ensemble import ExtraTreesClassifier
# Building the model
extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy', max_features = 2)
extra_tree_forest.fit(X_train, y_train)

predictions_e = extra_tree_forest.predict(X_test)
print('Extra_tree_forest Accuracy: ', accuracy_score(y_test, predictions_e)*100)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=250,shuffle=True)
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.get_params()
gradient_booster.fit(X_train,y_train)
predictions_g = gradient_booster.predict(X_test)
print('gradient_booster Accuracy: ', accuracy_score(y_test, predictions_g)*100)

pickle.dump(gradient_booster, open('model.pkl','wb'))