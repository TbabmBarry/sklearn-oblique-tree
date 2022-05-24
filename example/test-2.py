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
    penguins_data = pd.read_csv('penguins.csv')
    # Preprocess
    penguins_data.dropna(inplace=True)
    penguins_data.drop(penguins_data.index[penguins_data['sex'] == '.'].tolist(),inplace=True)
    penguins_data['species']=penguins_data['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})
    penguins_data.reset_index(inplace=True)
    penguins_data.drop('index', axis=1, inplace=True)
    df_male = pd.get_dummies(data=penguins_data['sex'], drop_first=True)
    penguins_data = pd.concat([penguins_data,df_male], axis=1)
    df_island = pd.get_dummies(data=penguins_data['island'])
    penguins_data = pd.concat([penguins_data,df_island], axis=1)
    pgn = penguins_data.drop(['island','sex'], axis=1)
    

    X = pgn.drop('species', axis=1)
    y = pgn['species']

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def get_results(X_train, y_train, X_test, y_test, classifiers):
    names = []
    tree_depth_list = []
    accuracy_list = []
    traditional_tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
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
    oblique_tree = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=random_state)
    # bivariate_tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=5, random_state=random_state)
    classifiers = {
        "OC1": oblique_tree, 
        # "Bivariate": bivariate_tree
    }
    results = get_results(X_train, y_train, X_test, y_test, classifiers)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(results)