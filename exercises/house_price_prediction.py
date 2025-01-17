from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).drop_duplicates().dropna()

    # Drop not needed features
    df.drop(['id'], axis=1, inplace=True)
    df.drop(['lat'], axis=1, inplace=True)
    df.drop(['long'], axis=1, inplace=True)

    # Edit date to datetime pandas
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%dT%f", errors='coerce')

    # only positive numbers
    lst = ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built",
           "sqft_living15", "sqft_lot15", "bathrooms",
           "floors"]
    for feature in lst:
        df = df[df[feature] > 0]
    # checks where there is a basement
    df['has_basement'] = np.where(df['sqft_basement'] > 0, 1, 0)
    # renovated in the last 10 years
    df['new_renovation'] = np.where(pd.DatetimeIndex(df['date']).year - df['yr_renovated'] < 10, 1, 0)
    # drop date
    df.drop(['date'], axis=1, inplace=True)

    # Edit Zip-code to dummies
    df = pd.get_dummies(df, columns=['zipcode'])

    # drop Nan to make sure
    df.dropna(axis=1, inplace=True)
    return df.drop("price", axis=1), df.price


def calc_pearson_correlation(feature, y):
    feature_st = np.std(feature)
    y_st = np.std(y)
    covariance = np.cov(feature, y)
    return covariance[0, 1] / (feature_st * y_st)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        name = feature.capitalize()
        pearsonCorrelation = calc_pearson_correlation(X[feature], y)
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}),
                         x="x",
                         y="y",
                         title=f"Correlation Between {name} feature values and "
                               f"response <br><sub> Pearson Correlation"
                               f" {pearsonCorrelation}",
                         labels={"x": f"{name} Values",
                                 "y": "Response Values"})
        pio.write_image(fig, output_path + "/" + name + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, y, r'C:\Users\moric\Documents\CS\year2\B\IML\projects\ex2\graphs')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, y,
                                                        train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    precent = np.linspace(10, 100, 91)
    mean_pred, var_pred = np.array([]), np.array([])
    train = train_X.join(train_y)

    for p in precent:
        lst = np.array([])
        for test in range(10):
            p_data = train.sample(frac=p / 100)
            t_x, t_y = p_data.drop("price", axis=1), p_data.price
            estimator = LinearRegression()
            estimator.fit(t_x.to_numpy(), t_y.to_numpy())
            loss = estimator.loss(test_X.to_numpy(), test_y.to_numpy())
            lst = np.append(lst, loss)

        mean_pred = np.append(mean_pred, np.mean(lst))
        var_pred = np.append(var_pred, np.std(lst))

    fig = go.Figure(
        (go.Scatter(x=precent, y=mean_pred, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"),
                    marker=dict(color="green", opacity=.7),
                    ),
         go.Scatter(x=precent, y=mean_pred - (2 * var_pred), fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=precent, y=mean_pred + 2 * var_pred, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),))

    st = "Linear regression model over increasing percentages of the " \
         "training set and the loss over the test set"
    fig.update_layout(title=st+"<br><sup>",
                      xaxis={'title': 'Percentage'},
                      yaxis={'title': 'MSE over ten iterations'}, title_x=0.5, title_font_size=15)
    fig.show()