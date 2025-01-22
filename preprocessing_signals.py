import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import itertools
import pywt
import random


def filter_faults(signal: np.ndarray, faults: np.ndarray, h: float = 0.001) -> np.ndarray:
    """
    Returns an array of faults by threshold

    Returns an array of faults by threshold

    :param signal: signal of wave
    :param faults: array of faults
    :param h: threshold
    :return: returns an array of faults
    """

    faults_mean = np.absolute(signal[faults])[0]
    arr = np.array([])
    for i in range(len(faults)):
        if abs(abs(signal[faults[i]]) - faults_mean) > h:
            arr = np.append(arr, faults[i])

    return arr


def detect_faults(signal: np.ndarray, min_threshold: float = 0.1, max_threshold: float = 25) -> tuple:
    """
    Returns an array of peaks by threshold

    Returns an array of peaks by threshold

    :param signal: signal of wave
    :param min_threshold: min threshold
    :param max_threshold: max threshold
    :return: returns an array of peaks
    """

    peaks = scipy.signal.find_peaks(abs(signal), prominence=(min_threshold, max_threshold))
    return peaks


def intervals_errors(signal: np.ndarray,
                     wavelet: str = 'db20',
                     level: int = 3,
                     epsilon: int = 2,
                     min_threshold: float = 0.1,
                     max_threshold: float = 25,
                     h: float = 1e-6) -> list:
    """
    Generate list of intervals for errors

    :param signal: array of signal wave
    :param wavelet: wavelet for wavelet decomposition
    :param level: level of decomposition
    :param epsilon: epsilon
    :param min_threshold: min threshold
    :param max_threshold: max threshold
    :param h: h
    :return: list of intervals
    """

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    cf_lvl = coeffs[level]
    peaks = detect_faults(cf_lvl, min_threshold, max_threshold)
    faults_index = filter_faults(cf_lvl, peaks[0], h)
    eps_peaks = [peak - epsilon for peak in peaks[0]]

    intervals = list(itertools.combinations(eps_peaks, 2))

    return intervals


def clean_signal(signal: np.ndarray) -> np.ndarray:
    """
    Function for cleaning signal

    :param signal: signal of x and y values
    :return: new signal of y
    """

    vals, counts = np.unique(signal["y"], return_counts=True)
    Ampl = vals[np.argmax(counts)]
    y = signal["y"] - abs(Ampl) * np.sin(100 * np.pi * signal["x"])

    return y


def vis_errors_intervals(y: np.ndarray, intervals: list) -> None:
    """
    Visualize plot of signal with errors intervals

    :param y: signal
    :param intervals: intervals of errors
    """

    for err in intervals:
        plt.plot(y)
        color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        plt.axvspan((err[0] * 2), (err[1] * 2), alpha=0.3, color=color, label='Выделенный интервал')
        plt.show()
