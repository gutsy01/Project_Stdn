import numpy as np


class MultinomialNBC:
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.feature_prob = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_prior = np.zeros(n_classes)
        self.feature_prob = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[i] = X_c.shape[0] / X.shape[0]
            self.feature_prob[i] = (X_c.sum(axis=0) + 1) / \
                (np.sum(X_c.sum(axis=0)) + n_features)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            posteriors = []
            for j, c in enumerate(self.classes):
                prior = np.log(self.class_prior[j])
                likelihood = np.sum(np.log(self.feature_prob[j]) * x)
                posterior = prior + likelihood
                posteriors.append(posterior)

            y_pred[i] = self.classes[np.argmax(posteriors)]
        return y_pred
