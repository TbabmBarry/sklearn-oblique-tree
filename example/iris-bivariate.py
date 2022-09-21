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


iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.ascontiguousarray(X)
tree.fit(X, y)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
coefs = tree.get_coef_arr(X.shape[1])
output = tree.write_tree()

print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
# print("Node coefficients:\n", coefs)
print("Oblique tree:\n", output)