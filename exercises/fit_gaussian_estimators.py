from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    amount = 1000
    X = np.random.normal(mu, sigma, amount)
    prob = UnivariateGaussian()
    prob.fit(X)
    print(prob.mu_, prob.var_)

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


def func(x, y, sigma, X):
    mu = np.array([x, 0, y, 0])
    return MultivariateGaussian.log_likelihood(X, mu, sigma)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    amount = 1000
    X = np.random.multivariate_normal(mu, sigma, amount)
    prob_multi = MultivariateGaussian()
    prob_multi.fit(X)
    print(prob_multi.mu_)
    print(prob_multi.cov_)

    # Question 5 - Likelihood evaluation
    amount = 200
    xaxis = np.linspace(-10, 10, amount)
    yaxis = np.linspace(-10, 10, amount)
    func_vec = np.vectorize(lambda x, y: func(x, y, sigma, X))
    result = func_vec(xaxis[:, np.newaxis], yaxis)

    labels = {"x": "f3", "y": "f1", "color": "log-Likelihood"}

    fig = px.imshow(result,
                    labels=labels,
                    x=xaxis,
                    y=yaxis)
    fig.update_layout(title_text="Q5- Multivariate Gaussian Estimator<br><sub>"
                                 "Log-Likelihood of expectation [f1,0,f3,0] "
                                 " while f1,f2 are values from -10 to 10 with "
                                 "samples from expectation [0,0,4,0]</br></sub>",
                      title_x=0.5)
    fig.show()

    # Question 6 - Maximum likelihood
    max_elemnt = np.where(result == np.amax(result))
    print("f1-"+str(np.around(xaxis[max_elemnt[0]],3)))
    print("f3-"+str(np.around(yaxis[max_elemnt[1]],3)))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

