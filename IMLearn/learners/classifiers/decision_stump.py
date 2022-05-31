from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # find the best split
        best_loss, best_j, best_sign, best_threshold = np.inf, None, None, None
        for j in range(X.shape[1]):
            for sign in [-1, 1]:
                threshold_new, loss = self._find_threshold(X[:, j], y, sign)
                if loss < best_loss:
                    best_loss, self.threshold_, self.j_, self.sign_ = loss, threshold_new, j, sign
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = values.shape[0]
        sort_by = np.argsort(values)
        values, labels = values[sort_by], labels[sort_by]
        thr = values[0]
        thr_err = np.sum(np.where(np.sign(labels) != sign, np.abs(labels), 0))
        thr_max = thr_err
        for i in range(1, n_samples):
            thr_new = values[i]
            thr_err += (labels[i - 1]) * sign
            if thr_err < thr_max:
                thr, thr_max = thr_new, thr_err
        return thr, thr_max / n_samples

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
        # return loss_functions.misclassification_error(y, self.predict(X))
        pred = self.predict(X)
        return np.sum(np.where(np.sign(y) != np.sign(pred), np.abs(pred), 0))


# labels = np.array([-1, -1, -1, 1, -1, -1, 1, 1, 1, 1])
# values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# print(values.shape)
# print(values)
# d = DecisionStump()
# print(d._find_threshold(values, labels, -1))

# loss = d._loss(values, labels)
# print(loss)

