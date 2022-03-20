from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    amount = 1000
    X = np.random.normal(mu, sigma, amount)
    prob = UnivariateGaussian()
    prob.fit(X)
    print(prob.mu_,prob.var_)


    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int)
    graph = []
    for m in ms:
        current = X[:m]
        calc = np.absolute(np.subtract(np.mean(current), mu))
        graph.append(calc)

    go.Figure([go.Scatter(x=ms, y=graph, mode='markers+lines', name=r'abs distance$')],
              layout=go.Layout(title=r"$\text{Q2 Absolute difference between estimated and true value of expectation "
                                     r"as a Function Of Number Of Samples}$",
                               xaxis_title="number of samples",
                               yaxis_title="deviation in expectation ")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_graph = prob.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf_graph, mode='markers', name=r'abs distance$')],
              layout=go.Layout(title=r"$\text{Q3 PDF as a function of the Samples}$",
                               xaxis_title="sample value",
                               yaxis_title="PDF")).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([[0], [0], [4], [0]])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0]])
    X = np.random.multivariate_normal

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
