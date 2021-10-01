import numpy as np


class AdaptiveFilter():
    """ 


    Creates an Adaptive Filter from a list of FIR Coefficients.

    Example:
    --------
    F = AdaptiveFilter([1,1,1])

    Methods:
    --------

    adapt_LMS
    adapt_NLMS 
    adapt_Newton_LMS

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

    def adapt_LMS(self, desired_signal: np.ndarray, input_signal: np.ndarray, step: float = 1e-2) -> dict:
        """
        Fit filter parameters to considering desired vector and inp
        ut x. desired and x must have length K,
        where K is the number of iterations

        Inputs
        -------

        desired : numpy array (row vector)
        desired signal
        x : numpy array (row vector)
        input signal to feed filter
        step : Convergence (relaxation) factor.

        Outputs
        -------

        python dict :
        outputs : numpy array (collumn vector)
        Store the estimated output of each iteration. outpu
        ts_vector[k] represents the output erros at iteration k
        errors : numpy array (collumn vector)
        FIR error vectors. error_vector[k] represents the o
        utput erros at iteration k.
        coefficients : numpy array
        Store the estimated coefficients for each iteration
        """
        max_iter = desired_signal.size

        x_k = np.zeros(self.filter_order+1, dtype=input_signal.dtype)

        errors_vector = np.array([])
        outputs_vector = np.array([])

        for k in range(max_iter):

            x_k = np.concatenate(([input_signal[k]], x_k))[
                :self.filter_order+1]

            w_k = self.coefficients
            y_k = np.dot(w_k.conj(), x_k)

            error_k = desired_signal[k] - y_k

            next_w_k = w_k + step * error_k.conj() * x_k

            errors_vector = np.append(errors_vector, error_k)
            outputs_vector = np.append(outputs_vector, y_k)
            #print(self.coefficients, [next_w_k])
            self.coefficients = next_w_k
            self.coefficients_history.append([next_w_k])
        # updates the current filter coefficients

        return {'outputs': outputs_vector,
                'errors': errors_vector, 'coefficients': self.coefficients}

    def adapt_NLMS(self, desired_signal: np.ndarray, input_signal: np.ndarray, gamma: float, step: float = 1e-2) -> dict:
        """
        Fit filter parameters to considering desired vector and inp
        ut x. desired and x must have length K,
        where K is the number of iterations

        Inputs
        -------

        desired : numpy array (row vector)
        desired signal
        x : numpy array (row vector)
        input signal to feed filter
        step : Convergence (relaxation) factor.

        Outputs
        -------

        python dict :
        outputs : numpy array (collumn vector)
        Store the estimated output of each iteration. outpu
        ts_vector[k] represents the output erros at iteration k
        errors : numpy array (collumn vector)
        FIR error vectors. error_vector[k] represents the o
        utput erros at iteration k.
        coefficients : numpy array
        Store the estimated coefficients for each iteration
        """
        max_iter = desired_signal.size

        x_k = np.zeros(self.filter_order+1, dtype=input_signal.dtype)

        errors_vector = np.array([])
        outputs_vector = np.array([])

        for k in range(max_iter):

            x_k = np.concatenate(([input_signal[k]], x_k))[
                :self.filter_order+1]

            w_k = self.coefficients[k]
            y_k = np.dot(w_k.conj(), x_k)

            error_k = desired_signal[k] - y_k
            gamma_f = step/(x_k.conj()*x_k + gamma)
            next_w_k = w_k + x_k * error_k.conj() * gamma_f

            next_w_k = w_k + step * error_k.conj() * x_k

            errors_vector = np.append(errors_vector, error_k)
            outputs_vector = np.append(outputs_vector, y_k)
            self.coefficients = np.concatenate(
                (self.coefficients, [next_w_k]))

        # updates the current filter coefficients
        self.coefficients = w_k
        self.coefficients_history.append([w_k])

        return {'outputs': outputs_vector,
                'errors': errors_vector, 'coefficients': self.coefficients}

    def adapt_Newton_LMS(self, desired_signal: np.ndarray, input_signal: np.ndarray, alpha: float, initialInvRxHat: np.ndarray, step: float = 1e-2) -> dict:
        """
        Fit filter parameters to considering desired vector and inp
        ut x. desired and x must have length K,
        where K is the number of iterations

        Inputs
        -------

        desired : numpy array (row vector)
        desired signal
        x : numpy array (row vector)
        input signal to feed filter
        step : Convergence (relaxation) factor.

        Outputs
        -------

        python dict :
        outputs : numpy array (collumn vector)
        Store the estimated output of each iteration. outpu
        ts_vector[k] represents the output erros at iteration k
        errors : numpy array (collumn vector)
        FIR error vectors. error_vector[k] represents the o
        utput erros at iteration k.
        coefficients : numpy array
        Store the estimated coefficients for each iteration
        """
        max_iter = desired_signal.size

        x_k = np.zeros(self.filter_order+1, dtype=input_signal.dtype)

        errors_vector = np.array([])
        outputs_vector = np.array([])

        for k in range(max_iter):

            x_k = np.concatenate(([input_signal[k]], x_k))[
                :self.filter_order+1]

            w_k = self.coefficients[k]
            y_k = np.dot(w_k.conj(), x_k)

            error_k = desired_signal[k] - y_k

            auxDen = (1-alpha)/alpha + x_k.conj()*invRxHat*x_k
            invRxHat = (invRxHat-(invRxHat*x_k*x_k.conj()
                                  * invRxHat)/auxDen)/(1 - alpha)
            next_w_k = w_k + x_k * error_k.conj() * step * invRxHat

            errors_vector = np.append(errors_vector, error_k)
            outputs_vector = np.append(outputs_vector, y_k)
            self.coefficients = np.concatenate(
                (self.coefficients, [next_w_k]))

        # updates the current filter coefficients
        self.coefficients = w_k
        self.coefficients_history.append([w_k])

        return {'outputs': outputs_vector,
                'errors': errors_vector, 'coefficients': self.coefficients}
