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

tree = ObliqueTree(splitter="bivariate", number_of_restarts=20, max_perturbations=10, random_state=random_state)

def dataPreprocessing():
    # Loading dataset
    seeds_data = pd.read_csv("seeds.csv")
    
    seeds_data.columns=['Area', 'Perimeter', 'Compactness', 'Kernel_Length','Kernel_Width', 'Asymmetry_Coeff','Kernel_Groove' , 'Category']
    seeds_data['Category'] = seeds_data['Category'].map({1:0, 2:1, 3:2})

    y = seeds_data['Category']
    X = seeds_data.iloc[:, 0:7]
    X = np.ascontiguousarray(X)
    return X, y

X, y = dataPreprocessing()

tree.fit(X, y)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
output = tree.write_tree()

print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
print("Oblique tree:\n", output)
