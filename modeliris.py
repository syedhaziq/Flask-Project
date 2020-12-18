from sklearn import datasets
import pandas as pd
import pickle
import numpy as np
iris = datasets.load_iris()


df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris['target']


def func(x):
    if x == 0:
        return "setosa"
    elif x==1:
        return "versicolor"
    elif x==2:
        return "virginica"

df['target_name']=df.target.apply(func)

x= df.iloc[: , 0:4]
y=df.iloc[: , 4]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn import svm

clf = svm.SVC()

clf.fit(X_train, y_train)


with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)