import pandas as pd 
import numpy as np 

random_state = 2

# Libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Libraries for ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn_oblique_tree.oblique import ObliqueTree

from sklearn.metrics import accuracy_score

# tree = ObliqueTree(splitter="bivariate", number_of_restarts=20, max_perturbations=10, random_state=random_state)

def dataPreprocessing():
    # Loading dataset
    diabetes_data = pd.read_csv("diabetes.csv")
    diabetes_data.drop(['Insulin'], axis=1, inplace=True)
    diabetes_data = diabetes_data[diabetes_data['DiabetesPedigreeFunction'] < 1.6]

    y = diabetes_data['Outcome']
    X = diabetes_data.drop(['Outcome'], axis=1)
    X = np.ascontiguousarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=40)
    # Data normalization
    Min_max_scaler = MinMaxScaler().fit(X_train)

    ## Scaling 
    X_train_mm_scaled = Min_max_scaler.transform(X_train)
    X_test_mm_scaled = Min_max_scaler.transform(X_test)
    return X_train_mm_scaled, X_test_mm_scaled, y_train, y_test

X_train, X_test, y_train, y_test = dataPreprocessing()

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

tree_depth = clf.get_depth()

# print("Iris Accuracy:",accuracy_score(y_test, predictions))
print("Diabetes Accuracy:",accuracy_score(y_test, predictions))
print("Tree depth:", tree_depth)