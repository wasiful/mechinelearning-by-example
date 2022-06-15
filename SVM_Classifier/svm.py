from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Wisconsin dataset, loading the data and basic analysis
cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
print('Input data size:', X.shape)
print('Output data size:', Y.shape)
print('label names:', cancer_data.target_names)
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples')
# always check whether classes are imbalanced before trying to solve any classification problem.

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# apply the SVM classifier to the data, kernel parameter set to linear, penalty hyperparameter C set to the
# default value, 1.0
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)  # fit the model
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is:{accuracy*100:.1f}%')

# b_cancer_data = load_breast_cancer()
# X = b_cancer_data.data
# Y = b_cancer_data.target
# print('Input data size:', X.shape)
# print('Output data size:', Y.shape)
# print('Label names:', b_cancer_data.target_names)

n_class0 = (Y == 0).sum()
n_class1 = (Y == 1).sum()
n_class2 = (Y == 2).sum()
print(f'{n_class0} class 0 samples,\n{n_class1} class1 samples, \n{n_class2} class2 samples.')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# apply the SVM classifier to the data
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)

# predict on the testing set with the trained model
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is:{accuracy*100:.1f}%')

# check how it performs for individual classes
pred = clf.predict(X_test)
print(classification_report(Y_test, pred))

# apply the RBF kernel with different values to a toy dataset
X = np.c_[
    # negative class
    (.3, -.8),
    (-1.5, -1),
    (-1.3, -.8),
    (-1.1, -1.3),
    (-1.2, -.3),
    (-1.3, -.5),
    (-.6, 1.1),
    (-1.4, 2.2),
    (1, 1),
    # positive class
    (1.3, .8),
    (1.2, .5),
    (.2, -2),
    (.5, -2.4),
    (.2, -2.3),
    (0, -2.7),
    (1.3, 2.1)
].T
Y = [-1] * 8 + [1] * 8
# take three values, 1 , 2 , and 4 , for kernel coefficient options
gamma_option = [1, 2, 4]
for i, gamma in enumerate(gamma_option, 1):
    svm = SVC(kernel="rbf", gamma=gamma)
    svm.fit(X, Y)  # fit the model

    # visualize the trained decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=['b']*8+['r']*8, zorder=10, cmap=plt.cm.Paired)
    plt.axis('tight')
    XX, YY = np.mgrid[-3:3:200j, -3:3:200j]
    Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.title('gamma = %d' % gamma)
    plt.show()
# a larger γ results in narrow regions
