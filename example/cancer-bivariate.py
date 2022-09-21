import pandas as pd 
import numpy as np 

random_state = 2

# Libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.metrics import accuracy_score

tree = ObliqueTree(splitter="bivariate", number_of_restarts=20, max_perturbations=10, random_state=random_state)

def dataPreprocessing():
    # Loading dataset
    cancer_data = pd.read_csv("cancer.csv")
    # Drop Useless column
    cancer_data.drop(['Unnamed: 32','id'], axis =1, inplace = True)
    # Reassign target
    cancer_data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)

    y = cancer_data.diagnosis
    X = cancer_data.drop('diagnosis', axis=1)

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.ascontiguousarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = dataPreprocessing()

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
output = tree.write_tree()

# print("Iris Accuracy:",accuracy_score(y_test, predictions))
print("Cancer Accuracy:",accuracy_score(y_test, predictions))
print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
print("Oblique tree:\n", output)