import typing

import numpy as np
import scipy as scp


def generalizedVandermondeMatrix(x: np.ndarray, functions: typing.List[typing.Callable]) -> np.ndarray:
    n, k = x.shape[0], len(functions)
    print(k)
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

    epsilon = np.random.normal(loc=0.0, scale=var ** 0.5, size=(n, ))
    y = phi_x @ theta_true + epsilon

    return x, y
