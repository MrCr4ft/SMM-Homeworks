import typing

import numpy as np
import scipy as scp


def reachedConvergence(x: np.ndarray, x_estimate: np.ndarray, grad_x0: np.ndarray,
                       grad_x: np.ndarray, tolf: float, tolx: float) -> bool:
    if np.linalg.norm(x_estimate - x, ord=2) < tolx:
        return True

    if np.linalg.norm(grad_x, ord=2) < tolf * np.linalg.norm(grad_x0, ord=2):
        return True

    return False


def backtracking(f, grad_f, x):
    """
    This function is a simple implementation of the backtracking algorithm for
    the GD (Gradient Descent) method.

    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    x: ndarray. The actual iterate x_k.
    """
    alpha = 1
    c = 0.8
    tau = 0.25

    while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.linalg.norm(grad_f(x), 2) ** 2:
        alpha = tau * alpha

        if alpha < 1e-3:
            break
    return alpha


def gradientDescent(f_pars_0: np.ndarray, f: typing.Callable, f_gradient: typing.Callable,
                    alpha: float = 1e-3, kmax: int = 500, tolf: float = 1e-6, tolx: float = 1e-6,
                    use_backtracking: bool = False, verbose: bool = False):
    f_pars_estimates = np.empty((kmax + 1,) + f_pars_0.shape, dtype=np.float32)
    f_pars_estimates[0] = f_pars_0

    f_vals = np.empty((kmax,), dtype=np.float32)
    grads = np.empty((kmax,) + f_pars_0.shape, dtype=np.float32)
    errs = np.empty((kmax,), dtype=np.float32)

    for i in range(kmax):
        if (i + 1) % 100 == 0 and verbose:
            print("Running iteration %d" % (i + 1))
        f_vals[i] = f(f_pars_estimates[i])
        grads[i] = f_gradient(f_pars_estimates[i])
        errs[i] = np.linalg.norm(grads[i], ord=2)

        if use_backtracking:
            alpha = backtracking(f, f_gradient, f_pars_estimates[i, :])

        f_pars_estimates[i + 1] = f_pars_estimates[i, :] - alpha * grads[i]
        if reachedConvergence(f_pars_estimates[i + 1, :], f_pars_estimates[i, :], grads[0], grads[i], tolf, tolx):
            return f_pars_estimates[:i + 2, :], i + 1, f_vals[:i + 2], grads[:i + 2, :], errs[:i + 1]

    return f_pars_estimates, None, f_vals, grads, errs


def stochasticGradientDescent(f_pars_0: np.ndarray, f: typing.Callable, f_gradient: typing.Callable,
                              data: typing.Tuple[np.ndarray, np.ndarray], alpha: float = 1e-3, batch_size: int = 32,
                              n_epochs: int = 100, verbose: bool = False):
    n = data[0].shape[0]
    n_batchs = n // batch_size
    indexes = np.arange(n)

    f_pars_estimates = np.zeros((n_epochs + 1,) + f_pars_0.shape, dtype=np.float32)
    f_pars_estimates[0] = f_pars_0
    f_vals = np.zeros((n_epochs,), dtype=np.float32)
    grads = np.zeros((n_epochs,) + f_pars_0.shape, dtype=np.float32)
    errs = np.zeros((n_epochs,), dtype=np.float32)

    for i in range(n_epochs):
        if (i + 1) % 100 == 0 and verbose:
            print("Running epoch %d..." % (i + 1))
        np.random.shuffle(indexes)
        grad = np.zeros(f_pars_0.shape)
        for mb in range(n_batchs):
            mb_idxs = indexes[mb * batch_size: (mb + 1) * batch_size]
            grad += f_gradient(data, f_pars_estimates[i])

        f_vals[i] = f(data, f_pars_estimates[i])
        grads[i] = grad
        errs[i] = np.linalg.norm(grad, ord=2)
        f_pars_estimates[i + 1] = f_pars_estimates[i, :] - alpha * grad

    return f_pars_estimates, f_vals, grads, errs


def generalizedVandermondeMatrix(x: np.ndarray, functions: typing.List[typing.Callable]) -> np.ndarray:
    n, k = x.shape[0], len(functions)
    phi_x = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            phi_x[i, j] = functions[j](x[i])

    return phi_x


def classicalVandermondeMatrix(x: np.ndarray, k: int) -> np.ndarray:
    functions = [
        (lambda exp: (lambda value: value ** exp))(i) for i in range(k)
    ]

    return generalizedVandermondeMatrix(x, functions)


def testProblem(k: int, n: int = 100, a: float = 0, b: float = 1, var: float = 1, theta_true: np.ndarray = None,
                vandermonde_functions: typing.List[typing.Callable] = None) -> typing.Tuple[np.ndarray, np.ndarray]:
    assert k > 0, "k must be strictly positive"
    assert var > 0, "The variance must be strictly positive"
    assert a < b, "The range given for X is invalid"

    if theta_true is None:
        theta_true = np.ones(k)
    else:
        assert theta_true.shape[0] == k, "Theta must have be of size k"

    x = np.linspace(start=a, stop=b, num=n)

    if vandermonde_functions is not None:
        phi_x = generalizedVandermondeMatrix(x, vandermonde_functions)
    else:
        phi_x = classicalVandermondeMatrix(x, k)

    epsilon = np.random.normal(loc=0.0, scale=var ** 0.5, size=(n,))
    y = phi_x @ theta_true + epsilon

    return x, y


def averageAbsoluteError(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = y_true.shape[0]
    return np.sum(np.abs(y_true - y_pred), axis=0).reshape(-1) / n


def polynomialRegressorError(test_set: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray) -> float:
    x_test, y_test = test_set
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x_test, k)
    f_theta = phi_x @ theta.reshape(-1, 1)
    return averageAbsoluteError(y_test, f_theta)


def MLEPolynomlialRegressionLoss(data: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray,
                                 sigma_estimate: float = 1.0):
    x, y = data
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x, k)
    f_theta = phi_x @ theta

    return 1 / (2 * sigma_estimate ** 2) * np.linalg.norm((f_theta - y), ord=2) ** 2


def MLEPolynomlialRegressionLossGradient(data: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray,
                                         sigma_estimate: float = 1.0):
    x, y = data
    n = x.shape[0]
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x, k)
    f_theta = phi_x @ theta

    return (1 / (sigma_estimate ** 2)) * phi_x.T @ (f_theta - y) / n


def MAPPolynomlialRegressionLoss(data: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray,
                                 regularization_parameter: float = 1.0):
    x, y = data
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x, k)
    f_theta = phi_x @ theta

    return np.linalg.norm((f_theta - y), ord=2) ** 2 / 2 + \
           (regularization_parameter / 2) * np.linalg.norm(theta, ord=2) ** 2


def MAPPolynomlialRegressionLossGradient(data: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray,
                                         regularization_parameter: float = 1.0):
    x, y = data
    n = x.shape[0]
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x, k)
    f_theta = phi_x @ theta

    return (phi_x.T @ (f_theta - y) / n) + (regularization_parameter * theta) / n


def MLEPolynomialRegression(data: typing.Tuple[np.ndarray, np.ndarray], k: int, sigma_estimate: float,
                            solving_strategy: str = "exact", alpha: float = 1e-3, kmax: int = 5000) -> np.ndarray:
    assert solving_strategy in ["exact", "gd", "sgd"], \
        "Solving strategy must be one among \"exact\", \"gd\", and \"sgd\""

    x, y = data
    phi_x = classicalVandermondeMatrix(x, k)

    if solving_strategy == "exact":
        theta = np.linalg.inv(phi_x.T @ phi_x) @ phi_x.T @ y

        return theta

    elif solving_strategy == "gd":
        theta = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        loss = lambda t: MLEPolynomlialRegressionLoss(data, t, sigma_estimate)
        loss_gradient = lambda t: MLEPolynomlialRegressionLossGradient(data, t, sigma_estimate)
        theta_estimates, k, f_vals, grads, errs = gradientDescent(theta, loss, loss_gradient, kmax=kmax,
                                                                  verbose=True, use_backtracking=True)

        return theta_estimates[-1]

    else:
        theta = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        loss = lambda d, t: MLEPolynomlialRegressionLoss(d, t, sigma_estimate)
        loss_gradient = lambda d, t: MLEPolynomlialRegressionLossGradient(d, t, sigma_estimate)
        theta_estimates, f_vals, grads, errs = stochasticGradientDescent(f_pars_0=theta, f=loss,
                                                                         f_gradient=loss_gradient,
                                                                         data=(x, y), alpha=alpha, batch_size=100,
                                                                         n_epochs=kmax, verbose=True)

        return theta_estimates[-1]


def MAPPolynomialRegression(data: typing.Tuple[np.ndarray, np.ndarray], k: int, regularization_parameter: float,
                            solving_strategy: str = "exact", alpha: float = 1e-3, kmax: int = 100) -> np.ndarray:
    assert solving_strategy in ["exact", "gd", "sgd"], \
        "Solving strategy must be one among \"exact\", \"gd\", and \"sgd\""

    x, y = data
    phi_x = classicalVandermondeMatrix(x, k)

    if solving_strategy == "exact":
        theta = np.linalg.inv(phi_x.T @ phi_x + regularization_parameter * np.eye(k)) @ phi_x.T @ y

        return theta

    elif solving_strategy == "gd":
        theta = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        loss = lambda t: MAPPolynomlialRegressionLoss(data, t, regularization_parameter)
        loss_gradient = lambda t: MAPPolynomlialRegressionLossGradient(data, t, regularization_parameter)
        theta_estimates, k, f_vals, grads, errs = gradientDescent(theta, loss, loss_gradient, kmax=kmax,
                                                                  verbose=True, use_backtracking=True)

        return theta_estimates[-1]

    else:
        theta = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        loss = lambda d, t: MAPPolynomlialRegressionLoss(d, t, regularization_parameter)
        loss_gradient = lambda d, t: MAPPolynomlialRegressionLossGradient(d, t, regularization_parameter)
        theta_estimates, f_vals, grads, errs = stochasticGradientDescent(f_pars_0=theta, f=loss,
                                                                         f_gradient=loss_gradient,
                                                                         data=(x, y), alpha=alpha, batch_size=100,
                                                                         n_epochs=kmax, verbose=True)

        return theta_estimates[-1]


def testProblemPoisson(k: int, n: int = 100, a: float = 0, b: float = 1,
                       theta_true: np.ndarray = None) -> typing.Tuple[np.ndarray, np.ndarray]:
    assert k > 0, "k must be strictly positive"
    assert a < b, "The range given for X is invalid"

    if theta_true is None:
        theta_true = np.ones(k)
    else:
        assert theta_true.shape[0] == k, "Theta must have be of size k"

    x = np.linspace(start=a, stop=b, num=n)
    phi_x = classicalVandermondeMatrix(x, k)

    y = phi_x @ theta_true
    epsilon = np.random.poisson(y, y.shape[0])

    return x, y + epsilon


def MLEPolynomlialRegressionPoissonLoss(data: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray):
    x, y = data
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x, k)
    f_theta = phi_x @ theta

    return np.sum(f_theta - y * np.log(f_theta), axis=0) * 1 / x.shape[0]


def MLEPolynomlialRegressionPoissonLossGradient(data: typing.Tuple[np.ndarray, np.ndarray], theta: np.ndarray):
    x, y = data
    k = theta.shape[0]
    phi_x = classicalVandermondeMatrix(x, k)

    return np.sum(phi_x - (y.reshape((-1, 1)) / theta.reshape((1, -1))), axis=0) / x.shape[0]


def MLEPolynomialRegressionPoisson(data: typing.Tuple[np.ndarray, np.ndarray], k: int,
                                   solving_strategy: str = "exact", alpha: float = 1e-3,
                                   kmax: int = 5000) -> np.ndarray:
    assert solving_strategy in ["exact", "gd", "sgd"], \
        "Solving strategy must be one among \"exact\", \"gd\", and \"sgd\""

    x, y = data
    phi_x = classicalVandermondeMatrix(x, k)

    if solving_strategy == "exact":
        theta = np.linalg.inv(phi_x.T @ phi_x) @ phi_x.T @ y * 0.5

        return theta

    elif solving_strategy == "gd":
        theta = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        loss = lambda t: MLEPolynomlialRegressionPoissonLoss(data, t)
        loss_gradient = lambda t: MLEPolynomlialRegressionPoissonLossGradient(data, t)
        theta_estimates, k, f_vals, grads, errs = gradientDescent(theta, loss, loss_gradient, kmax=kmax,
                                                                  verbose=True, use_backtracking=True)

        return theta_estimates[-1] / 2

    else:
        theta = np.random.normal(loc=0.0, scale=1.0, size=(k,))
        loss = lambda d, t: MLEPolynomlialRegressionPoissonLoss(d, t)
        loss_gradient = lambda d, t: MLEPolynomlialRegressionPoissonLossGradient(d, t)
        theta_estimates, f_vals, grads, errs = stochasticGradientDescent(f_pars_0=theta, f=loss,
                                                                         f_gradient=loss_gradient,
                                                                         data=(x, y), alpha=alpha, batch_size=100,
                                                                         n_epochs=kmax, verbose=True)

        return theta_estimates[-1] / 2
