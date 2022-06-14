import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

data_path = 'datastore/movie-rating-data/ratings.csv'
n_users = 6040
n_movies = 9724


def load_rating_data(data_path, n_users, n_movies):
    """
    :param data_path: path of data file
    :param n_users: number of users
    :param n_movies: number of movies with ratings
    :return: rating data of numpy array[user, movie]
    """
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split(",")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            try:
                rating = float(rating)
            except ValueError as e:
                raise ValueError(f"Expected number got {rating}, Trace: {e}")
            internal_movie_id = movie_id_mapping[movie_id]
            print(f"Adding movie id {movie_id} to user id {user_id} with internal user id {internal_movie_id}")
            if internal_movie_id < n_movies:
                data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0.0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping

# load_rating_data(data_path, n_users, n_movies)


t_data, t_movie_n_rating, t_movie_id_mapping = load_rating_data(data_path, n_users, n_movies)


def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of ratings {int(value)}:{count}')


display_distribution(t_data)

movie_id_most, n_rating_most = sorted(t_movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f'Movie id {movie_id_most} has {n_rating_most} ratings')


X_raw = np.delete(t_data, t_movie_id_mapping[movie_id_most], axis=1)
Y_raw = t_data[:, t_movie_id_mapping[movie_id_most]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
print('shape of X:', X.shape)
print('shape of X:', Y.shape)
display_distribution(Y)

recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(len(Y_train), len(Y_test))


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])

prediction = clf.predict(X_test)
print(prediction[:10])

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0,1]))

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test, prediction, pos_label=1)
recall_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=0)

from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)

pos_prob = prediction_prob[:, 1]
thresholds = np. arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] += 1
        else:
            break

n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()

true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]


import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)
plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characterestics')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, pos_prob)

from sklearn.model_selection import StratifiedKFold
k = 5
k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X, Y):
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_train = Y[train_indices], Y[test_indices]
    for  alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train, Y_train)
                prediction_prob = clf.predict_proba(X_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)


for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.item():
        print(f'{smoothing}     {fit_prior}     {auc/k:.5f}')

clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the best model:', roc_auc_score(Y_test, pos_prob))
# sparce matrix
"""
        movie1, movie2, movie2
user1   4       5       
user2           5       5
user3   3       4       5

Numpy Representation
[
    [4, 5, 0]
    [0, 5, 5]
    [3, 4, 5]
]
"""
