import numpy as np
from sklearn.naive_bayes import BernoulliNB

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 0],
    [1, 1, 0]
])
Y_train = ['Y', 'N', 'Y', 'Y']

X_test = np.array([[1, 1, 0]])


clf = BernoulliNB(alpha=1.0, fit_prior=True)  # initing model with smoothing and prior
clf.fit(X_train, Y_train)  # to train with fit method

pred_prob = clf.predict_proba(X_test)  # probability results with the predict_proba method
print('scitlrn Predicted probabilities:\n', pred_prob)
pred = clf.predict(X_test)  # directly acquire the predicted class with the predict method
print('scik-learn prediction:', pred)

