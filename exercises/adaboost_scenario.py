import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    number_fitted_learners = np.arange(1, n_learners + 1)
    train_errors , test_errors = [], []
    for i in number_fitted_learners:
        train_errors.append(adaboost.partial_loss(train_X, train_y, i))
        test_errors.append(adaboost.partial_loss(test_X, test_y, i))
    fig = go.Figure([go.Scatter(x=number_fitted_learners, y=train_errors, mode="lines", name="Train error")])
    fig.add_scatter(x=number_fitted_learners, y=test_errors, mode="lines", name="Test error")
    fig.update_layout(title_text=f"Train and test errors of AdaBoost model (data with {noise} noise)",
                      xaxis_title="Number of fitted base learners",
                      yaxis_title="Error")
    fig.show()

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:
    # Question 2: Plotting decision surfaces
        T = [5, 50, 100, 250]
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Number of Learners: {num}" for num in T],
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        for i, t in enumerate(T):
            fig.add_traces(
                [decision_surface(lambda samples: adaboost.partial_predict(samples, t), lims[0], lims[1], showscale=False),
                 go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                            marker=dict(color=test_y.astype(int), symbol=class_symbols[test_y.astype(int)],
                                        colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))],
                rows=[(i // 2) + 1,(i // 2) + 1] , cols=[(i % 2) + 1,(i % 2) + 1])
        fig.update_layout(title=rf"$\textbf{{Decision Boundaries After 5, 50, 100, 250 Iterations}}$",
                             margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()



        # Question 3: Decision surface of best performing ensemble
        num_learners = np.argmin(test_errors) + 1
        accuracy = 1 - test_errors[num_learners - 1]
        fig = go.Figure([decision_surface(lambda samples: adaboost.partial_predict(samples, num_learners)
                        , lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]], line=dict(width=1)))])

        fig.update_layout(title=f"Decision Boundaries by Adaboost <br>"
                                f"Lowest Test Error With {num_learners} Base Learners ({noise} noise)<br>"
                                f"Accuracy: {accuracy}",
                          margin=dict(t=110), title_x=0.5, title_font_size=20, width=800)
        fig.update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()

    # Question 4: Decision surface with weighted samples
    D = (adaboost.D_ / np.max(adaboost.D_)) * 5
    go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y.astype(int), symbol="circle",
                                      colorscale=[custom[0], custom[-1]],
                                      line=dict(color="black", width=1), size=D))]).update_layout(
        title="The Training set with a point size proportional to it’s weight").update_xaxes(visible=False).update_yaxes(visible=False).show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
