import numpy as np
from collections import defaultdict


class NaiveBayes:
    def __init__(self, feat_vec: np.array, category_list: np.array):
        self.X = feat_vec
        self.y = category_list
        self.label_indices = None
        self.likelihood = None
        self.prior = None
        self.posterior = None

    def get_label_indices(self) -> defaultdict:
        """
        :return: dict,{class1:[indices], class2: [indices]}
        """
        label_indices = defaultdict(list)
        for index, label in enumerate(self.y):
            label_indices[label].append(index)
        self.label_indices = label_indices
        return label_indices

    def get_prior(self) -> dict:
        """
        :return: dictionary, with class label as key, corresponding prior as value
        """
        if self.label_indices is None:
            self.get_label_indices()
        prior = {label: len(indices) for label, indices in self.label_indices.items()}
        # alternative long method
        # prior = dict()
        # for label, indices in label_indices.items():
        #     prior[label] = len(indices)
        total_count = sum(prior.values())
        for label in prior:
            prior[label] /= total_count
        return prior

    def get_likelihood(self, smoothing=0.0):
        """
        :param smoothing: intiger additive smoothing parameter
        :return: dict, with class key, conditional probability
        """
        likelihood = {}
        label_indices = self.get_label_indices()
        for label, indices in label_indices.items():
            likelihood[label] = self.X[indices, :].sum(axis=0) + smoothing
            total_count = len(indices)
            likelihood[label] = likelihood[label]/(total_count + 2 * smoothing)
        return likelihood

    def get_posterior(self, X: np.array):
        """

        :param X: testing samples
        :return: dictionary,  with class label key, posterior value
        """
        posteriors = []
        for x in X:
            posterior = self.prior.copy()
            for label, likelihood_label in self.likelihood.items():
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

    def fit(self, smoothing=1.0):
        likelihood = self.get_likelihood(smoothing)
        prior = self.get_prior()
        self.likelihood = likelihood
        self.prior = prior

        posterior = self.get_posterior(self.X)
        self.posterior = posterior
        return self

    def predict(self, feature: np.array) -> list:
        if self.prior is None or self.likelihood is None:
            raise ValueError("The model has not been trained yet! Please train the model first.")
        prediction = self.get_posterior(feature)
        return prediction


if __name__ == "__main__":
    X_train = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [1, 1, 0]
    ])
    Y_train = ['Y', 'N', 'Y', 'Y']

    nbm = NaiveBayes(feat_vec=X_train, category_list=Y_train)
    nbm = nbm.fit()
    nbm.predict(np.array([[1, 1, 0]]))
    X_test = np.array([[1, 1, 0]])

    print('label_indices:\n', nbm.label_indices)
    print('Likelihood:\n', nbm.likelihood)
    print('prior:', nbm.prior)
    print('posterior:\n', nbm.posterior)


