import numpy as np
from collections import defaultdict

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 0],
    [1, 1, 0]
])
Y_train = ['Y', 'N', 'Y', 'Y']

X_test = np.array([[1, 1, 0]])


def get_label_indices(labels: list) -> defaultdict:
    """
    :param labels: list of labels
    :return: dict,{class1:[indices], class2: [indices]}
    """
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


test_label_indices = get_label_indices(Y_train)
print('label_indices:\n', test_label_indices)


def get_prior(label_indices: defaultdict) -> dict:
    """
    :param label_indices:grouped sample indices by class
    :return: dictionary, with class label as key, corresponding prior as value
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    # alternative long method
    # prior = dict()
    # for label, indices in label_indices.items():
    #     prior[label] = len(indices)
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior


test_prior = get_prior(test_label_indices)
print('prior:', test_prior)


def get_likelihood(features, label_indices, smoothing=0):
    """
    :param features: matrix of features
    :param label_indices: grouped sample indices by class
    :param smoothing: intiger additive smoothing parameter
    :return: dict, with class key, conditional probability
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label]/(total_count + 2 * smoothing)
    return likelihood


test_smoothing = 1
t_likelihood = get_likelihood(X_train, test_label_indices, test_smoothing)
print('Likelihood:\n', t_likelihood)


def get_posterior(X, prior, likelihood):
    """

    :param X: testing samples
    :param prior: dictionary, with class label key
    :param likelihood: dictionary, with classs label key
    :return: dictionary,  with class label key, posterior value
    """
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1-likelihood_label[index])
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


posterior = get_posterior(X_test, test_prior, t_likelihood)
print('posterior:\n', posterior)


