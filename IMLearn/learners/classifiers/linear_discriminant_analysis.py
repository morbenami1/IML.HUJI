from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics import loss_functions



class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.shape[0], n_features))
        self.cov_ = np.zeros((n_features, n_features))
        self.pi_ = np.zeros((self.classes_.shape[0]))
        for i in range(self.classes_.shape[0]):
            self.pi_[i] = np.sum(y == self.classes_[i]) / n_samples
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
        for i in range(n_samples):
            a = np.array([X[i] - self.mu_[y[i]]])
            self.cov_ += a.T @ a
        self.cov_ /= n_samples
        self._cov_inv = inv(self.cov_)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        responses = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            responses[i] = self.classes_[np.argmax(likelihood[i])]
        return responses



    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i in range(self.classes_.shape[0]):
            exp = np.exp(-0.5 * np.sum((X - self.mu_[i]) @ self._cov_inv * (X - self.mu_[i]) , axis=1))
            sqr = np.sqrt(np.power(2 * np.pi, X.shape[1]) * det(self.cov_))
            likelihoods[:, i] = self.pi_[i] * exp / sqr
        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_predict = self._predict(X)
        return loss_functions.misclassification_error(y, y_predict)

