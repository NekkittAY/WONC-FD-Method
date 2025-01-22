from joblib import Parallel, delayed
import numpy as np
import scipy
from math_module import objective_function
from math_module import pad_array


def optimize_params(i: int,
                    size: int,
                    shift: int,
                    y: np.ndarray,
                    err: np.ndarray,
                    method: str,
                    bounds: list,
                    maxfev: int) -> float:
    """
    Optimize function for parallelize

    :param i: number of process
    :param size: size of function
    :param shift: shifts of function
    :param y: array of signal
    :param err: error to classify
    :param method: method of optimization
    :param bounds: bounds of function for optimization
    :param maxfev: max iterations
    :return: result of function after optimization
    """

    init_params = [size, shift]
    res = scipy.optimize.minimize(objective_function,
                                  init_params,
                                  args=(y, err),
                                  method=method,
                                  bounds=bounds,
                                  options={'maxfev': maxfev})

    return res.fun


def WoncFD_classification_parallel(y: np.ndarray,
                                 errors: list = ["haar", "haar1", "exp", "-exp", "Invexp", "Invexp+", "algb"],
                                 leng: int = 1000,
                                 epsilon: float = 0.01,
                                 n: float = 0.98,
                                 method: str = 'Nelder-Mead',
                                 maxfev: int = 3000,
                                 iters: int = 20) -> list:
    """
    WONC-FD classification parallel

    :param y: dots of signal with error
    :param errors: list of errors
    :param leng: length of y
    :param epsilon: small number to check proximity to 0
    :param n: percentage of the size of the array with 0 to ignore it
    :param method: method of optimization
    :param maxfev: max iterations
    :return: array with class of function and MSE
    """

    if len(y) <= leng * 0.05 or sum(1 for point in y if abs(point) < epsilon) > len(y) * n:
        return ["Bad signal", "Nan"]

    y = pad_array(y, leng)
    result = np.array([])
    bounds = [(len(y) - 10, len(y) + 10), (-len(y), len(y))]

    size_values = np.arange(len(y) - 10, len(y) + 10, 1)
    shifts = np.arange(-len(y), len(y), 1)

    selected_indices_size = np.linspace(0, len(size_values) - 1, iters).astype(int)
    selected_indices_shift = np.linspace(0, len(shifts) - 1, iters).astype(int)

    size = size_values[selected_indices_size]
    shift = shifts[selected_indices_shift]

    for err in errors:
        best_losses = Parallel(n_jobs=-1)(delayed(optimize_params)(i, size[i], shift[i], y, err, method, bounds, maxfev) for i in range(iters))
        best_loss = np.min(best_losses)
        result = np.append(result, best_loss)

    #result = [errors[np.argmin(result)], result[np.argmin(result)]]
    result = [[errors[err], result[err]] for err in range(len(result))]

    return result
