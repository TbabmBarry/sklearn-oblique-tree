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

tree = DecisionTreeClassifier()

def dataPreprocessing():
    # Loading dataset
    heart_data = pd.read_csv('cleveland.csv', header = None)

    heart_data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang', 
                'oldpeak', 'slope', 'ca', 'thal', 'target']

    # Preprocess
    heart_data['target'] = heart_data.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    heart_data['thal'] = heart_data.thal.fillna(heart_data.thal.mean())
    heart_data['ca'] = heart_data.ca.fillna(heart_data.ca.mean())

    X = heart_data.iloc[:, :-1].values
    y = heart_data.iloc[:, -1].values

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = dataPreprocessing()

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

tree_depth = tree.get_depth()

# print("Iris Accuracy:",accuracy_score(y_test, predictions))
print("Heart Accuracy:",accuracy_score(y_test, predictions))
print("Tree depth:", tree_depth)
