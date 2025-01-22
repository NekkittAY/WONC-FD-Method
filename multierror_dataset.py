from math import *
import numpy as np
import pandas as pd


def intervals_zeros(arr: np.ndarray,
                    intervals: list = None,
                    num: int = 0) -> list:
    """
    Returns a binned array

    Splits an array into intervals up to points,
    which are specified in intervals or into n parts,
    which are specified in num

    :param arr: source array to split into intervals
    :param intervals: an array of intervals to which you want to split
    :param num: the number of intervals to split into
    :return: Returns a binned array
    """

    if not (num == 0):
        res = [arr[d: d + num] for d in range(0, len(arr), num)]
    else:
        res = []
        intervals = sorted(intervals)
        temp = []
        k = 0
        for i in range(len(arr)):
            if (arr[i] < intervals[k]) or (arr[i] > intervals[k] and k == len(intervals) - 1):
                temp.append(arr[i])
            else:
                temp.append(arr[i])
                res.append(temp)
                temp = []
                if k + 1 < len(intervals):
                    k += 1
        res.append(temp)

    result = []

    for i in range(len(res)):
        temp = np.array(res[i])
        if len(temp) < len(arr):
            lenght = sum([len(j) for j in res[:i]])
            zeros = np.zeros(lenght)
            if len(zeros) > 0:
                temp = np.concatenate((zeros, temp))
            zeros = np.zeros(len(arr) - len(temp))
            if len(zeros) > 0:
                temp = np.concatenate((temp, zeros))
        result.append(temp)

    arr = result

    return arr


def haar(x: np.ndarray,
         y: np.ndarray,
         a: float,
         b: float,
         c: float,
         k: float) -> list:
    """
    Returns the haar function of f(x) and x

    Returns the haar function with the specified function height from -k to k
    on the interval from a to b and from b to c,
    changing the arrays of the definition area - f(x) and the range of values - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: start of first interval
    :param b: the end of the first and the beginning of the second interval
    :param c: end of second interval
    :param k: haar magnitude factor in y
    :return: returns the haar function of f(x) - y and x
    """

    result = []

    for i in range(len(x)):
        if a <= x[i] < b:
            result.append(k)
        elif b <= x[i] < c:
            result.append(-k)
        else:
            result.append(y[i])

    return result


def haar_1(x: np.ndarray,
           y: np.ndarray,
           a: float,
           b: float,
           k: float) -> list:
    """
    Returns single step haar function of f(x) and x

    Returns a single-stage haar function with the specified function height from -k to k
    on the interval from a to b,
    changing the arrays of the definition area - f(x) - y and the range of values - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: start of interval
    :param b: end of the interval
    :param k: haar magnitude factor in y
    :return: returns single step haar function of f(x) and x
    """

    result = []

    for i in range(len(x)):
        if a <= x[i] <= b:
            result.append(k)
        else:
            result.append(y[i])

    return result


def error_exp(x: np.ndarray,
              y: np.ndarray,
              a: float,
              b: float,
              t: float = 0,
              delta: float = 0.2,
              epsilon: float = 0.0001) -> list:
    """
    Returns exponent over interval

    Returns an exponential error function on the interval from a to b,
    changing the range array - f(x) - y and the range array - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: start of first interval
    :param b: the end of the second interval
    :param t: exponent attribute
    :param delta: exponent attribute increment step
    :param epsilon: value to compare two float values
    :return: returns exponent over interval
    """

    result = []

    for i in range(len(x)):
        if a <= x[i] < b:
            result.append(exp(t - 1) / 100)
            t += delta
        elif abs(x[i] - b) <= epsilon:
            result.append(0)
        else:
            result.append(y[i])

    return result


def error_minus_exp(x: np.ndarray,
                    y: np.ndarray,
                    a: float,
                    b: float,
                    c: float,
                    k: float,
                    t: float = 0,
                    delta: float = 0.15,
                    epsilon: float = 0.0001) -> list:
    """
    Returns minus exponent over interval

    Returns the error function of the opposite exponent
    over the interval from b to c with a splash in a, c
    hanging the range array - f(x) - y and the range array - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: start of first interval
    :param b: the end of the first and the beginning of the second interval
    :param c: the end of the second interval
    :param k: error magnitude factor in y
    :param t: exponent attribute
    :param delta: exponent attribute increment step
    :param epsilon: value to compare two float values
    :return: returns minus exponent over interval
    """

    result = []

    for i in range(len(x)):
        if b <= x[i] <= c:
            result.append(-exp(t) / 400)
            t += delta
        elif abs(x[i] - a) <= epsilon:
            result.append(k)
        else:
            result.append(y[i])

    return result


def error_inv_exp(x: np.ndarray,
                  y: np.ndarray,
                  a: float,
                  b: float,
                  c: float,
                  k: float,
                  t: float = 0,
                  delta: float = 0.1,
                  epsilon: float = 0.0001) -> list:
    """
    Returns the inverse exponent over an interval

    Returns the inverse exponential error function
    from b to c with a splash in a,
    changing the range array - f(x) - y and the range array - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: start of first interval
    :param b: the end of the first and the beginning of the second interval
    :param c: the end of the second interval
    :param k: error magnitude factor in y
    :param t: exponent attribute
    :param delta: exponent attribute increment step
    :param epsilon: value to compare two float values
    :return: returns the inverse exponent over an interval
    """

    result = []

    for i in range(len(x)):
        if b <= x[i] <= c:
            result.append(1 / exp(t - 2))
            t += delta
        elif abs(x[i] - a) <= epsilon:
            result.append(k)
        else:
            result.append(y[i])

    return result


def error_inv_exp_plus(x: np.ndarray,
                       y: np.ndarray,
                       a: float,
                       b: float,
                       c: float,
                       k: float,
                       t: float = 0,
                       delta: float = 0.1,
                       epsilon: float = 0.0001) -> list:
    """
    Returns the inverse exponent with a shift up over an interval

    Returns the inverse exponential error function with a shift
    from b to c with a splash in a,
    changing the range array - f(x) - y and the range array - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: start of first interval
    :param b: the end of the first and the beginning of the second interval
    :param c: the end of the second interval
    :param k: error magnitude factor in y
    :param t: exponent attribute
    :param delta: exponent attribute increment step
    :param epsilon: value to compare two float values
    :return: returns the inverse exponent with a shift up over an interval
    """

    result = []

    for i in range(len(x)):
        if b <= x[i] <= c:
            result.append((-exp(t) + 500) / 100)
            t += delta
        elif abs(x[i] - a) <= epsilon:
            result.append(k)
        else:
            result.append(y[i])

    return result


def algb_3(x: np.ndarray,
           y: np.ndarray,
           a: float,
           b: float,
           c: float,
           d: float,
           i0: float,
           i1: float,
           t: float = -1.7,
           delta: float = 0.03) -> list:
    """
    Returns a polynomial of degree 3 over an interval

    Returns a polynomial of degree 3
    from i0 to i1 withwith coefficients a,b,c,d,
    changing the range array - f(x) - y and the range array - x

    :param x: initial array x
    :param y: initial array f(x)
    :param a: first polynomial coefficient
    :param b: second polynomial coefficient
    :param c: third polynomial coefficient
    :param d: fourth polynomial coefficient
    :param i0: start of first interval
    :param i1: end of the interval
    :param t: polynomial attribute
    :param delta: polynomial attribute increment step
    :return: returns a polynomial of degree 3 over an interval
    """

    result = []

    for i in range(len(x)):
        if i0 <= x[i] <= i1:
            result.append(a * t ** 3 + b * t ** 2 + c * t + d)
            t += delta
        else:
            result.append(y[i])

    return result


def transform(x: np.ndarray,
              y: np.ndarray,
              sx: float,
              sy: float) -> np.ndarray:
    """
    Returns the transformed matrix

    Returns the x-axis shrink or stretch matrix
    (with the x-axis stretch factor - sx)
    and y (with the y-axis stretch factor - sy)

    :param x: initial array x
    :param y: initial array y
    :param sx: compression/expansion ratio x
    :param sy: compression/expansion ratio y
    :return: returns the transformed matrix
    """

    y = np.array([x, y])
    transform_matrix = np.array([[sx, 0], [0, sy]])
    y_transformed = np.dot(transform_matrix, y)

    return y_transformed


def Error_signal(signal: str,
                 x: np.ndarray,
                 a: float,
                 t: float) -> np.ndarray:
    """
    Returns an array of y given the function and stretch factors

    Returns an array with the selected error function
    from the list of errors with amplitude factor (y-axis stretch)
    and time stretch (x-axis stretch)

    :param signal: error input function
    :param x: initial array x
    :param a: compression/expansion ratio amplitude (y)
    :param t: compression/expansion ratio time (x)
    :return: returns an array of y given the function and stretch factors
    """

    y = np.zeros(len(x))
    if signal == "haar":
        y = haar(x, y, 0.075, 0.1, 0.125, 3)
    elif signal == "haar1":
        y = haar_1(x, y, 0.05, 0.15, 3)
    elif signal == "exp":
        y = error_exp(x, y, 0.065, 0.1)
    elif signal == "-exp":
        y = error_minus_exp(x, y, 0.09, 0.1, 0.15, 0)
    elif signal == "Invexp":
        y = error_inv_exp(x, y, 0.09, 0.1, 0.15, 0.7)
    elif signal == "Invexp+":
        y = error_inv_exp_plus(x, y, 0.09, 0.1, 0.15, 0.7)
    elif signal == "algb":
        y = algb_3(x, y, 1, 1, 1, 0, 0.05, 0.15)

    x_arr, y_arr = np.array(transform(x, y, t, a), dtype=np.float32)

    return y_arr


def create_multierror_dataset(quan_in: int,
                              power_quan_in: int,
                              time_in: float = 1,
                              errors: list = ["haar", "haar1", "exp", "-exp", "Invexp", "Invexp+", "algb"],
                              signal_in: str = "sin") -> pd.DataFrame:
    """
    Returns a dataset of y and x for the specified number of failures and different signal amplitudes

    Returns a DataFrame over time_in with the number of random errors quan_in,
    the amount of random amplitude of the base signal (based on a sine wave) power_quan_in
    and with errors from the errors list

    :param quan_in: number of failures
    :param power_quan_in: the number of signals of different amplitudes
    :param time_in: signal time
    :param errors: set of errors
    :param signal_in: input signal (according to the standard - sinusoid)
    :return: returns a dataset of y and x for the specified number of failures and different signal amplitudes
    """

    x = np.arange(0, time_in, 0.001, dtype=np.float32)
    start = time_in / 1000

    t_quan = np.random.uniform(start, time_in - 0.2, quan_in)
    rand_err = np.random.choice(errors, quan_in)
    rand_max_ampl_pom = np.random.uniform(1, 25, quan_in)

    arr_errors = []
    for i, err in enumerate(rand_err):
        arr_error = np.concatenate((np.zeros(round(t_quan[i] / 0.001)), Error_signal(err, x, rand_max_ampl_pom[i], 1)))[
                    :len(x)]
        arr_errors.append(arr_error)

    arr_signal = np.array(intervals_zeros(x, num=round(len(x) / power_quan_in)))
    rand_max_ampl_signal = np.random.uniform(1, 25, power_quan_in)

    for i in range(power_quan_in):
        arr_signal[i] = transform(arr_signal[i], np.sin(100 * np.pi * arr_signal[i]), 1, rand_max_ampl_signal[i])[1]

    result_signal = np.sum(arr_signal, axis=0)
    result_errors = np.sum(arr_errors, axis=0)
    result = result_signal + result_errors

    dataset = {"x": x,
               "y": result}

    dataset = pd.DataFrame(dataset)

    return dataset
