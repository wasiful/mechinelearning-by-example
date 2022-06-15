import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# loading data fatal state classification in cardiotocography
df = pd.read_excel('datastore/CTG.xls', "Raw Data")

# Taking data sample and assigning feature sets
X = df.iloc[1:2126, 3:-2].values
Y = df.iloc[1:2126, -1].values
print(Counter(Y))  # checking class properties

# splitting and taking 20% data for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# RBF based svm model using penalty C and kernel coefficient gamma
svc = SVC(kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5), 'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}
grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)
print(grid_search.best_score_)

# Applying the optimal model to testing set
svc_best = grid_search.best_estimator_
accuracy = svc_best.score(X_test, Y_test)
print(f'The accuracy is:{accuracy*100:.1f}%')

# data is imbalanced, checking performance for individual classes
prediction = svc_best.predict(X_test)
report = classification_report(Y_test, prediction)
print(report)