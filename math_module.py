import numpy as np
import scipy
from multierror_dataset import Error_signal


def MSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE function

    :param y: array of dots
    :param y_pred: array of dots
    :return: MSE, float
    """

    return np.mean(np.square(y - y_pred))


def pad_array(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Function for pad an array

    :param arr: array
    :param length: length for new array
    :return: new array
    """

    current_len = len(arr)
    if current_len >= length:
        return arr
    else:
        pad_len = length - current_len
        left_len = pad_len // 2
        right_len = pad_len - left_len
        arr = np.pad(arr,
                     (left_len, right_len),
                     mode='constant',
                     constant_values=0)

    return arr


def resize_vector(vector: np.ndarray, new_size: int) -> np.ndarray:
    """
    Resize vector

    :param vector: array
    :param new_size: new size
    :return: array with new size
    """

    old_size = len(vector)

    if new_size == old_size:
        return vector

    if new_size > old_size:
        indices = np.linspace(0, old_size - 1, new_size, dtype=float)
        return np.interp(indices, np.arange(old_size), vector)

    indices = np.linspace(0, old_size - 1, new_size, dtype=float)
    resized_vector = np.zeros(new_size)

    for i in range(new_size):
        start_index = int(indices[i])
        end_index = start_index + 1 if i < new_size - 1 else old_size

        resized_vector[i] = np.mean(vector[start_index:end_index])

    return resized_vector


def WoncFD_MSE_classification(y: np.ndarray,
                       errors: list = ["haar", "haar1", "exp", "-exp", "Invexp", "Invexp+", "algb"],
                       leng: int = 1000,
                       epsilon: float = 0.01,
                       n: float = 0.98) -> list:
    """
    WONC-FD MSE classification

    :param y: dots of signal with error
    :param errors: list of errors
    :param leng: length of y
    :param epsilon: small number to check proximity to 0
    :param n: percentage of the size of the array with 0 to ignore it
    :return: array with class of function and MSE
    """

    if len(y) <= leng * 0.05 or sum(1 for point in y if abs(point) < epsilon) > len(y) * n:
        return ["Bad signal", "Nan"]

    result = np.array([])
    x = np.linspace(0, 1, len(y))
    ampl = 1
    size_values = np.arange(len(y), 2 * len(y), 10)
    shifts = np.arange(-len(y), len(y), 1)

    for err in errors:
        mse_err = 1e5
        for s in size_values:
            for shift in shifts:
                if shift >= 0:
                    Error = np.resize(resize_vector(Error_signal(err, x, ampl, 1), s), (1, len(y)))[0]
                    arr_error = np.concatenate((np.zeros(shift), Error))[:len(x)]
                else:
                    Error = np.resize(resize_vector(Error_signal(err, x, ampl, 1), s), (1, len(y)))[0]
                    arr_error = np.concatenate((Error, np.zeros(-shift)))[-len(x):]

                # Аппроксимация сигнала с использованием метода наименьших квадратов
                features_matrix = np.vstack((np.ones(len(x)), arr_error)).T
                coefficients, _, _, _ = np.linalg.lstsq(features_matrix, y, rcond=None)
                approximated_signal = np.dot(features_matrix, coefficients)

                mse = MSE(y, approximated_signal)
                if mse < mse_err:
                    mse_err = mse
        result = np.append(result, mse_err)

    result = [errors[np.argmin(result)], result[np.argmin(result)]]

    return result


def WoncFD_classification_brute(y: np.ndarray,
                              errors: list = ["haar", "haar1", "exp", "-exp", "Invexp", "Invexp+", "algb"],
                              leng: int = 1000,
                              epsilon: float = 0.01,
                              n: float = 0.98) -> list:
    """
    WONC-FD classification brute

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
    result = np.array([])
    x = np.linspace(0, 1, len(y))
    ampl = 1
    size_values = np.arange(len(y) - 10, len(y) + 10, 1)
    shifts = np.arange(-len(y), len(y), 1)

    for err in errors:
        mse_err = 1e5
        for s in size_values:
            for shift in shifts:
                Error = np.resize(resize_vector(Error_signal(err, x, ampl, 1), s), (1, len(y)))[0]
                if shift >= 0:
                    arr_error = np.concatenate((np.zeros(shift), Error))[:len(x)]
                else:
                    arr_error = np.concatenate((Error, np.zeros(-shift)))[-len(x):]

                # Аппроксимация сигнала с использованием метода наименьших квадратов
                features_matrix = np.vstack((np.ones(len(x)), arr_error)).T
                coefficients, _, _, _ = np.linalg.lstsq(features_matrix, y, rcond=None)
                approximated_signal = np.dot(features_matrix, coefficients)

                mse = MSE(y, approximated_signal)
                if mse < mse_err:
                    mse_err = mse
        result = np.append(result, mse_err)

    result = [errors[np.argmin(result)], result[np.argmin(result)]]

    return result


def objective_function(params: list, y: np.ndarray, error: str) -> float:
    """
    Objective function for minimization

    :param params: array of params
    :param y: dots of signal with error
    :param error: error
    :return: MSE of function with that params
    """

    size, shift = params
    size, shift = int(size), int(shift)

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


def WoncFD_classification(y: np.ndarray,
                        errors: list = ["haar", "haar1", "exp", "-exp", "Invexp", "Invexp+", "algb"],
                        leng: int = 1000,
                        epsilon: float = 0.01,
                        n: float = 0.98,
                        method: str = 'Nelder-Mead',
                        maxfev: int = 3000,
                        iters: int = 20) -> list:
    """
    WONC-FD classification

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
        best_loss = np.inf

        for i in range(iters):
            init_params = [size[i], shift[i]]
            res = scipy.optimize.minimize(objective_function,
                                          init_params,
                                          args=(y, err),
                                          method=method,
                                          bounds=bounds,
                                          options={'maxfev': maxfev})
            if res.fun < best_loss:
                best_loss = res.fun
        result = np.append(result, best_loss)

    #result = [errors[np.argmin(result)], result[np.argmin(result)]]
    result = [[errors[err], result[err]] for err in range(len(result))]

    return result
