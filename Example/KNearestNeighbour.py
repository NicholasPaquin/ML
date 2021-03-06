import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

df = pd.read_csv('datasets/breast-cancer-wisconsin.data.txt')
df.replace('?', -999999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measure = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 7, 9, 6, 5, 7, 5, 5, 8]])
example_measure = example_measure.reshape(len(example_measure), -1)

prediction = clf.predict(example_measure)
print(prediction)


