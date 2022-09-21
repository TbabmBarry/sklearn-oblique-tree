import pandas as pd 
import numpy as np 

random_state = 2

# Libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree = DecisionTreeClassifier()

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
    return X, y

X, y = dataPreprocessing()

tree.fit(X, y)

tree_depth = tree.get_depth()
print("Tree depth:", tree_depth)