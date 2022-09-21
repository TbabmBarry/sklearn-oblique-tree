import pandas as pd 
import numpy as np 

random_state = 2

from sklearn.datasets import load_iris, load_breast_cancer

# Libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.metrics import accuracy_score

tree = ObliqueTree(splitter="bivariate", number_of_restarts=10, max_perturbations=10, random_state=random_state)

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
    X = np.ascontiguousarray(X)
    return X, y

X, y = dataPreprocessing()

tree.fit(X, y)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
coefs = tree.get_coef_arr(X.shape[1])
output = tree.write_tree()
print("Penguins Dataset:")
print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
# print("Node coefficients:\n", coefs)
print("Oblique tree:\n", output)