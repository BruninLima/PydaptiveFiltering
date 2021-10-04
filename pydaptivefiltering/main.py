import numpy as np


class AdaptiveFilter():
    """ 


    Creates an Adaptive Filter from a list of FIR Coefficients.

    Example:
    --------
    F = AdaptiveFilter([1,1,1])

    Methods:
    -------

    filter_signal

    Auxiliar Methods:
    -----------------

    _reset_history
    _reset

    """

    def __init__(self, coefs):
        """ Main Class Docstring """

        array_coefs = coefs

        if type(array_coefs) != np.ndarray:
            array_coefs = np.array(array_coefs)

        self.coefficients = array_coefs
        self.filter_order = self.coefficients.size - 1
        self.coefficients_history = [array_coefs]

    def filter_signal(self, input_signal):
        """Returns a output signal"""

        input_array = input_signal

        if type(input_array) != np.ndarray:
            input_array = np.array(input_array)

        max_iter = input_array.size

        x_k = np.zeros(self.filter_order+1, dtype=input_array.dtype)
        y = np.zeros_like(input_signal)

        for k in range(max_iter):
            x_k = np.concatenate(([input_array[k]], x_k))[:-1]
            y[k] = np.dot(self.coefficients.conj(),  x_k)

        return y

    def _reset_history(self):
        "Auxiliar function to reset coefficients_history"
        self.coefficients_history = [self.coefficients]

    def _reset(self):
        self.coefficients = self.coefficients_history[0]
        self._reset_history()
