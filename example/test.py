from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=.4, random_state=random_state)

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

tree_depth = tree.get_depth()
node_cnt = tree.get_node_cnt()
coefs = tree.get_coef_arr(X_train.shape[1])
output = tree.write_tree()

print("Iris Accuracy:",accuracy_score(y_test, predictions))
print("Tree depth:", tree_depth)
print("Node count:", node_cnt)
print("Node coefficients:\n", coefs)
print("Oblique tree:\n", output)