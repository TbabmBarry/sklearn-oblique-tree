import pandas as pd 
import numpy as np 
import random
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')

# Libraries for preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Libraries for Classification models
from sklearn.tree import DecisionTreeClassifier
from sklearn_oblique_tree.oblique import ObliqueTree

random_state = 2

def dataPreprocessing():
    # Loading dataset
    data = pd.read_csv('column_3C_weka.csv')
    data['class'] = data['class'].map({'Hernia':0,'Spondylolisthesis':1,'Normal':2})
    
    X = data.drop('class', axis=1)
    y = data['class']

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.ascontiguousarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def get_results(X_train, y_train, X_test, y_test, classifiers):
    names = []
    tree_depth_list = []
    accuracy_list = []
    traditional_tree = DecisionTreeClassifier()
    classifiers["DT"] = traditional_tree
    for name, cls in classifiers.items():
        cls.fit(X_train, y_train)
        y_preds_test = cls.predict(X_test)
        accuracy = accuracy_score(y_test, y_preds_test)
        tree_depth = cls.get_depth()
        accuracy_list.append(accuracy)
        names.append(name)
        tree_depth_list.append(tree_depth)
    results = {
        "Models": names,
        'Accuracy': accuracy_list,
        'Tree Depth': tree_depth_list
    }
    resultsDF = pd.DataFrame.from_dict(results)
    
    return resultsDF


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = dataPreprocessing()
    # oblique_tree = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=random_state)
    bivariate_tree = ObliqueTree(splitter="bivariate", number_of_restarts=20, max_perturbations=5, random_state=random_state)
    classifiers = {
        # "OC1": oblique_tree, 
        "Bivariate": bivariate_tree
    }
    results = get_results(X_train, y_train, X_test, y_test, classifiers)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(results)