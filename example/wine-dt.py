import pandas as pd 
import numpy as np 

random_state = 2

# Libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn import tree

tree = DecisionTreeClassifier(max_depth=None)

dataset = load_wine()

X, y = dataset.data, dataset.target

tree.fit(X, y)

tree_depth = tree.get_depth()

tree_rules = export_text(tree, feature_names=dataset['feature_names'])

print("Tree depth:", tree_depth)
print("Decision tree:\n", tree_rules)
