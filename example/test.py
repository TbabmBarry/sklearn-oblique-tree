from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.tree import DecisionTreeClassifier

random_state = 2

#see Murthy, et all for details.
#For oblique with consideration of axis parallel
# tree = ObliqueTree(splitter="oc1, axis_parallel", number_of_restarts=20, max_perturbations=5, random_state=random_state)
#
#For multivariate CART select 'cart' splitter
tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=5, random_state=random_state)
# model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=10)

#consider only oblique splits
# tree = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=.4, random_state=random_state)

tree.fit(X_train, y_train)
# model_dt.fit(X_train, y_train)

predictions = tree.predict(X_test)
# predictions = model_dt.predict(X_test)

tree_depth = tree.get_depth()
# tree_depth = model_dt.get_depth()

# leaf_cnt = tree.leafCount()
# node_cnt = tree.nodeCount()

# coefs = tree.getCoef(X_train.shape[1])
tests = tree.test()

print("Iris Accuracy:",accuracy_score(y_test, predictions))
# print("Num of attr:", X_train.shape[1])
# print("Leaf count:", leaf_cnt)
print("Tree depth:", tree_depth)
# print("Node count:", node_cnt)
# print("Node coefficients", coefs)
for i in range(tests.shape[0]):
    print(tests[i])