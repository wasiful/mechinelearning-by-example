from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Labeled Faces in the Wild (LFW) people dataset
# load the face image data
face_data = fetch_lfw_people(min_faces_per_person=80) # classes with at least 80 samples

# Python Image Library(PIL) required for JPEG files
X = face_data.data
Y = face_data.target

# check amount of loaded data, analyze the label distribution
print('Input data size:', X.shape)
print('output data size:', Y.shape)
print(' Label names:', face_data.target_names)
for i in range(5):
    print(f'class {i} has {(Y == i).sum()} samples.')

# plotting few images
fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=face_data.target_names[face_data.target[i]])
    plt.show()

# number of dimensions is greater than the number of samples. SVM is effective at solving.
# Building the svm classifier
# split data for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
clf = SVC(class_weight='balanced', random_state=42)  # dataset is imbalanced, balance weight

# instead of cross validation by folding,  use grid search which handles entire process.
parameters = {'C': [0.1, 1, 10], 'gamma': [1e-07, 1e-08, 1e-06], 'kernel': ['rbf', 'linear']}
# initized grid search conducts 5fold cross validation and parallel, runs on all available cores 'job'
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)  # hyper parameter tuning by fit
print('The best model:\n', grid_search.best_params_)
print('the best 5fold average performance:', grid_search.best_score_)

# calculate the accuracy and classification report
clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)
print(f'The accuracy is:{clf_best.score(X_test, Y_test)*100:.1f}%')
print(classification_report(Y_test, pred, target_names=face_data.target_names))

# compressing the input features with principal component analysis
# PCA component projects the original data into a 100-dimension space
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)

# concatenate multiple consecutive steps and treat as one model. pipelining API.
# piplining two different modle, pca and svc
model = Pipeline([('pca', pca), ('svc', svc)])
parameters_pipeline = {'svc__C': [1, 3, 10], 'svc__gamma': [0.001, 0.005]}
# grid search for the best model
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)

print('The best model: \n', grid_search.best_params_)
print('The best averaged performance:', grid_search.best_score_)

model_best = grid_search.best_estimator_
print(f'The accuracy is: {model_best.score(X_test, Y_test)*100:.1f}%')
pred = model_best.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))

