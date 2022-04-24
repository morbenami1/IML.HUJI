import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"

Months_lst = ["January", "February", "March", "April", "May",
              "June", "July", "August", "September",
              "October", "November", "December"]
Months_dic = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
              6: "June", 7: "July", 8: "August", 9: "September",
              10: "October", 11: "November", 12: "December"}


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=["Date"],
                     dayfirst=True).drop_duplicates().dropna()
    df["DayOfYear"] = df['Date'].dt.dayofyear

    df.drop(['Date'], axis=1, inplace=True)
    df.drop(df[df.Temp < 0].index, inplace=True)

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # 1
    by_israel = df.drop(df[df.Country != "Israel"].index)

    years_by_order = list(sorted(by_israel["Year"].unique()))
    by_israel["Year"] = by_israel["Year"].astype(str)
    fig = px.scatter(by_israel, x="DayOfYear", y="Temp", color="Year",
                     category_orders={"Year": years_by_order},
                     color_discrete_sequence=px.colors.sequential.Agsunset)
    fig.update_layout(title="Average daily temperature change as a function"
                            " of the `DayOfYear`<br><sup>",
                      xaxis={'title': 'Day Of Year'},
                      yaxis={'title': 'Temperature'}, title_x=0.5,
                      title_font_size=15,
                      legend_title="Year")
    fig.show()

    # 2
    israel_month = by_israel.groupby('Month').agg(np.std)
    israel_month["Month"] = Months_lst
    fig = px.bar(israel_month, x='Month', y='Temp')
    fig.update_layout(title="The Standard Deviation of the Temperature as"
                            " a function of the Month"
                            "for Israel <br><sup>",
                      xaxis={'title': 'Month'},
                      yaxis={'title': 'Standard Deviation of Temperature'},
                      title_x=0.5, title_font_size=15)
    fig.show()

    # Question 3 - Exploring differences between countries
    country_month = df.groupby(['Month', 'Country']). \
        agg({"Temp": [np.mean, np.std]}).reset_index()
    country_month.columns = ["Month", "Country", "Mean", "Std"]
    country_month["Month"] = country_month["Month"].apply(lambda x:
                                                          Months_dic.get(x))
    fig = px.line(country_month, x="Month", y="Mean", color="Country",
                  error_y="Std", line_shape='spline')
    fig.update_layout(title="Average Monthly Temperature by Country",
                      xaxis={'title': 'Month'},
                      yaxis={'title': 'Temperature'}, title_x=0.5,
                      title_font_size=15,
                      legend_title="Country")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    lst_k = np.arange(1, 11, 1).tolist()
    lst_loss = []
    train_X, train_y, test_X, test_y = split_train_test(
        by_israel[["DayOfYear"]], by_israel["Temp"], train_proportion=0.75)
    for k in lst_k:
        estimator = PolynomialFitting(k)
        estimator.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = round(estimator.loss(test_X.to_numpy(),
                                    test_y.to_numpy()), 2)
        print(str(k) + ": " + str(loss))
        lst_loss.append(loss)

    k_dic = {'Degree': lst_k, 'Loss': lst_loss}
    df_deg = pd.DataFrame(k_dic)
    fig = px.bar(df_deg, x='Degree', y='Loss')
    fig.update_layout(title="For each degree in  "
                            "Polynomial Fitted The loss from"
                            " data in Israel <br><sup>",
                      xaxis={'title': 'Degree'},
                      yaxis={'title': 'Loss from MSE'},
                      title_x=0.5, title_font_size=15)
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    estimator = PolynomialFitting(4)
    estimator.fit(by_israel[["DayOfYear"]],by_israel["Temp"])
    df_without_israel = df.drop(df[df.Country == "Israel"].index)
    countries = list(sorted(df_without_israel["Country"].unique()))
    lst_loss_c = []
    for c in countries:
        by_c = df_without_israel.drop(
            df_without_israel[df_without_israel.Country != c].index)
        loss = estimator.loss(by_c[["DayOfYear"]], by_c["Temp"])
        lst_loss_c.append(loss)
    c_dic = {'Country': countries, 'Loss': lst_loss_c}
    df_countries = pd.DataFrame(c_dic)
    fig = px.bar(df_countries, x='Country', y='Loss')
    fig.update_layout(title="For Each Country The Loss With Polynomial Fitted by"
                            " Israel <br><sup>",
                      xaxis={'title': 'Country'},
                      yaxis={'title': 'Loss from MSE'},
                      title_x=0.5, title_font_size=15)
    fig.show()

