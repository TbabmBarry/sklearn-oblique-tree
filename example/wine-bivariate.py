import pandas as pd 
import numpy as np 

random_state = 2

# Libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

tree = ObliqueTree(splitter="bivariate", number_of_restarts=20, max_perturbations=10, random_state=random_state)

dataset = load_wine()

X, y = dataset.data, dataset.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.ascontiguousarray(X)
tree.fit(X, y)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
output = tree.write_tree()
print("Wine Dataset:")
print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
print("Oblique tree:\n", output)