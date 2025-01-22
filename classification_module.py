import numpy as np
import tensorflow as tf


def errors_classification(signal: np.ndarray,
                          interval: list,
                          threshold: float = 0.7,
                          model_path: str = 'Models/model_756.h5') -> list:
    """
    Errors classification

    :param signal: signal
    :param interval: interval of error
    :param threshold: threshold
    :param model_path: path to keras model
    :return: array of class of error in interval
    """

    model = tf.keras.models.load_model(model_path)
    errors = []
    sig_err = np.array([], dtype=np.float32)
    for index in range(len(signal)):
        if interval[0] <= index <= interval[1]:
            sig_err = np.append(sig_err, signal[index])
        else:
            sig_err = np.append(sig_err, 0)
    sig_err = sig_err.reshape(1, len(sig_err), 1)
    result = model.predict(sig_err)[0]
    if ((np.sum(result == 1) == 1 and np.sum(result == 0) == len(result) - 1) or
            result[np.argmax(result)] < threshold):
        return [signal, interval, result[np.argmax(result)], 0]
    else:
        return [signal, interval, result[np.argmax(result)], 1]
