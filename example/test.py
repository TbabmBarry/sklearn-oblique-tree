import pandas as pd 
import numpy as np 

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.tree import DecisionTreeClassifier

random_state = 2

# See Murthy, et all for details.
# For oblique with consideration of axis parallel
# tree = ObliqueTree(splitter="oc1, axis_parallel", number_of_restarts=20, max_perturbations=5, random_state=random_state)
#
# For multivariate CART select 'cart' splitter
# tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=10, random_state=random_state)

# Consider only oblique splits
# tree = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=random_state)

# For bivariate with consideration of axis parallel (default)
tree = ObliqueTree(splitter="bivariate", number_of_restarts=20, max_perturbations=10, random_state=random_state)

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
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X = np.ascontiguousarray(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=101)
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=.7, random_state=random_state)
X_train, X_test, y_train, y_test = dataPreprocessing()

# Export training dataset into csv files
# pd.DataFrame(X_train).to_csv("train_x_iris.csv", index=False)
# pd.DataFrame(y_train).to_csv("train_y_iris.csv", index=False)
pd.DataFrame(X_train).to_csv("train_x_penguins.csv", index=False, header=False)
pd.DataFrame(y_train).to_csv("train_y_penguins.csv", index=False, header=False)

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
coefs = tree.get_coef_arr(X_train.shape[1])
output = tree.write_tree()

# print("Iris Accuracy:",accuracy_score(y_test, predictions))
print("Penguins Accuracy:",accuracy_score(y_test, predictions))
print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
print("Node coefficients:\n", coefs)
print("Oblique tree:\n", output)