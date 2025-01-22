from joblib import Parallel, delayed
import numpy as np
import optuna
from math_module import pad_array, MSE, resize_vector
from multierror_dataset import Error_signal


def objective_function(trial, y: np.ndarray, error: str):
    """
    Objective function for optuna optimizer

    :param trial: optuna trial
    :param y: dots of signal with error
    :param error: error
    :return: MSE of function with that params
    """

    size = trial.suggest_int('size', len(y) - 10, len(y) + 10)
    shift = trial.suggest_int('shift', -len(y), len(y))

    x = np.linspace(0, 1, len(y))
    ampl = 1

    if shift >= 0:
        Error = np.resize(resize_vector(Error_signal(error, x, ampl, 1), size), (1, len(y)))[0]
        arr_error = np.concatenate((np.zeros(shift), Error))[:len(x)]
    else:
        Error = np.resize(resize_vector(Error_signal(error, x, ampl, 1), size), (1, len(y)))[0]
        arr_error = np.concatenate((Error, np.zeros(-shift)))[-len(x):]

    features_matrix = np.vstack((np.ones(len(x)), arr_error)).T
    coefficients, _, _, _ = np.linalg.lstsq(features_matrix, y, rcond=None)
    approximated_signal = np.dot(features_matrix, coefficients)

    mse = MSE(y, approximated_signal)
    return mse


def optimize_params(y: np.ndarray, error: str, iters: int):
    """
    Optimize params by optuna function

    :param y: dots of signal with error
    :param error: error
    :param iters: number of iterations for optimization
    :return: best value of optuna optimization
    """
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective_function(trial, y, error), n_trials=iters)
    return study.best_value


def WoncFD_optuna_classification_parallel(y: np.ndarray,
                                   errors: list = ["haar", "haar1", "exp", "-exp", "Invexp", "Invexp+", "algb"],
                                   leng: int = 1000,
                                   epsilon: float = 0.01,
                                   n: float = 0.98,
                                   iters: int = 100) -> list:
    """WONC-FD with Optuna classification parallel

    :param y: dots of signal with error
    :param errors: list of errors
    :param leng: length of y
    :param epsilon: small number to check proximity to 0
    :param n: percentage of the size of the array with 0 to ignore it
    :return: array with class of function and MSE
    """
    if len(y) <= leng * 0.05 or sum(1 for point in y if abs(point) < epsilon) > len(y) * n:
        return ["Bad signal", "Nan"]

    y = pad_array(y, leng)

    best_losses = Parallel(n_jobs=-1)(delayed(optimize_params)(y, error, iters) for error in errors)
    best_loss_idx = np.argmin(best_losses)
    best_error = errors[best_loss_idx]
    best_loss = best_losses[best_loss_idx]

    result = [best_error, best_loss]

    return result
