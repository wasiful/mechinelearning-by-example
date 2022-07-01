import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# fraction of positive class varies from 0 to 1
# linear spacing to take difference of 1, starting from 0 and 1000 points for smoother graph

pos_fraction = np.linspace(0.00, 1.00, 1000)

# pure split = leaf node
# Gini Impurity is calculated and plot of Gini Impurity versus Positive fraction
gini = 1 - pos_fraction**2 - (1-pos_fraction)**2  # eqn of gini impurity, 1-pos_fraction is the negative fraction

# in graph entropy increases with increasing probability of positive value
# max value for entropy is 1(50%) but gini impurity 0.5(50%), curve gets smaller and computationally efficient
plt.plot(pos_fraction, gini)
plt.ylim(0, 1)
plt.xlabel("Positive fraction")
plt.ylabel("Gini Impurity")
plt.show()


# giving labels of dataset to calculate gini impurity
def gini_impurity(labels):
    if not labels:  # empty set is pure
        return 0
    # count occurrence of each label
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions**2)


#  if a column has less gini impurity then other 3 columns so, its best to split on the lowest impure column
# giving list of 0 and 1 to check the gini impurity calculation
print(f'{gini_impurity([1, 1, 0, 1, 0]):.4f}')
print(f'{gini_impurity([1, 1, 0, 1, 0, 0]):.4f}')
print(f'{gini_impurity([1, 1, 1, 1]):.4f}')

# range of values for calculation
pos_fraction = np.linspace(0.00, 1.00, 1000)
# equation for calculating entropy
ent = - (pos_fraction * np.log2(pos_fraction) + (1 - pos_fraction)*np.log2(1-pos_fraction))
plt.plot(pos_fraction, ent)
plt.xlabel('Positive fraction')
plt.ylabel('Entropy')
plt.ylim(0, 1)
plt.show()


# the function goes through provided list to calculate entropy
def entropy(labels):
    if not labels:
        return 0
    # from a column or an array/list it takes all unique values and how many times they have occurred, here 0 or 1
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))  # amount of repeat / total amount , to calculate the probability
    return - np.sum(fractions * np.log2(fractions))  # equation of entropy is implemented


# checking function using some lists
print(f'{entropy([1, 1, 0, 1, 0]):.4f}')
print(f'{entropy([1, 1, 0, 1, 0, 0]):.4f}')
print(f'{entropy([1, 1, 1, 1]):.4f}')

criterion_function = {'gini': gini_impurity, 'entropy': entropy}  # initializing the 2 created functions
def weighted_impurity(groups, criterion='gini'):  # using the gini impurity func
    """
    :param groups: list of child, list of class label
    :param criterion: metric of mesure quality of split, gini Impurity, entropy for info gain
    :return: float, weighted impurity
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        new_result = len(group) / float(total) * criterion_function[criterion](group)
        weighted_sum = weighted_sum + new_result
        # weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum


children_1 = [[1, 0, 1], [0, 1]]
children_2 = [[1, 1], [0, 0, 1]]
print(f"Entropy of #1 split: {weighted_impurity(children_1, 'entropy'):.4f}")
print(f"Entropy of #2 split: {weighted_impurity(children_2, 'entropy'):.4f}")


def gini_impurity_np(labels):
    if labels.size == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)


def entropy_np(labels):
    if labels.size == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return - np.sum(fractions * np.log2(fractions))


criterion_function_np = {'gini': gini_impurity_np, 'entropy': entropy_np}
def weighted_impurity(groups, criterion='gini'):
    """
    :param groups: list of child, list of class label
    :param criterion: metric of mesure quality of split, gini Impurity, entropy for info gain
    :return: float, weighted impurity
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function_np[criterion](group)
    return weighted_sum


#  split a node into left and right children based on a feature and a value
def split_node(X, y, index, value):
    """
    :param X: np.ndarray, dataset feature
    :param y: np.ndarray, dataset target
    :param index: int, index of feature used for splitting
    :param value: value of the feature
    :return: list, list, left right child, child is in formate [x, y]
    """
    x_index = X[:, index]
    if X[0, index].dtype.kind in ['i', 'f']:    # if feature numeric
        mask = x_index >= value
    else:
        mask = x_index == value     # if feature is categorical
    left = [X[~mask, :], y[~mask]]  # split left and right child
    right = [X[mask, :], y[mask]]
    return left, right


# search function tries out all possible splits and returns the best one given a selection criterion
def get_best_split(X, y, criterion):
    """
    :param X: np.ndarray, dataset feature
    :param y: np.ndarray, dataset target
    :param criterion: gini impurity
    :return: index of feature, left right child
    """
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len(X[0])):
        print(index)
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return{'index': best_index, 'value': best_value, 'children': children}


def get_leaf(labels):
    return np.bincount(labels).argmax()


def split(node, max_depth, min_size, depth, criterion):
    left, right = node['children']
    del(node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1, criterion)

        if right[1].size <= min_size:
            node['right'] = get_leaf(right[1])
        else:
            result = get_best_split(right[0], right[1], criterion)
            result_left, result_right = result['children']
            if result_left[1].size == 0:
                node['right'] = get_leaf(result_right[1])
            elif result_right[1].size == 0:
                node['right'] = get_leaf(result_left[1])
            else:
                node['right'] = result
                split(node['right'], max_depth, min_size, depth+1, criterion)


def train_tree(X_train, y_train, max_depth, min_size, criterion='gini'):
    X = np.array(X_train)
    y = np.array(y_train)
    root = get_best_split(X, y, criterion)
    split(root, max_depth, min_size, 1, criterion)
    return root  # root


X_train = [['tech', 'professional'],
           ['fashion', 'student'],
           ['fashion', 'professional'],
           ['sports', 'student'],
           ['tech', 'student'],
           ['tech', 'retired'],
           ['sports', 'professional'],
           ]
y_train = [1, 0, 0, 0, 1, 0, 1]
tree = train_tree(X_train, y_train, 2, 2)
CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'is', 'no': 'is not'}}

get_best_split(np.array(X_train), np.array(y_train), "gini")


def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if node['value'].dtype.kind in ['i', 'f']:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
            print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
            print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth+1)
    else:
        print(f"{depth * ' '}[{node}]")


visualize_tree(tree)
X_train_n = [
    [6, 7],
    [2, 4],
    [7, 2],
    [3, 6],
    [4, 7],
    [5, 2],
    [1, 6],
    [2, 0],
    [6, 3],
    [4, 1]
]
y_train_n = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
tree = train_tree(X_train_n, y_train_n, 2, 2)
visualize_tree(tree)


tree_sk = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
tree_sk.fit(X_train_n, y_train_n)
export_graphviz(tree_sk, out_file='tree.dot', feature_names=['X1', 'X2'], impurity=False,
                filled=True, class_names=['0', '1'])
