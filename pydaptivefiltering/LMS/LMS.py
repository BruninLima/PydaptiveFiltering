import numpy as np
from time import time


def LMS(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, step: float = 1e-2, max_runs: int = 25, tolerance=0, verbose=(True, 5)) -> dict:
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

    max_runs
    max_iter

    verbose (boolean, int): (True/False, Print_every)


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
    verbose, print_every = verbose
    # assert type(verbose) == bool
    # assert type(print_every) == int && print_every > 0

    total_tic = time()
    tic = time()
    for run in range(max_runs):

        if verbose == True:
            if run == max_runs - 1 or run % print_every == 0:
                print("Run {}\t ".format(run), end='')

        max_iter = desired_signal.size

        x_k = np.zeros(Filter.filter_order+1, dtype=input_signal.dtype)

        errors_vector = np.array([])
        outputs_vector = np.array([])

        for k in range(max_iter):

            x_k = np.concatenate(([input_signal[k]], x_k))[
                :Filter.filter_order+1]

            w_k = Filter.coefficients
            y_k = np.dot(w_k.conj(), x_k)

            error_k = desired_signal[k] - y_k

            next_w_k = w_k + step * error_k.conj() * x_k

            errors_vector = np.append(errors_vector, error_k)
            outputs_vector = np.append(outputs_vector, y_k)

            Filter.coefficients = next_w_k
            Filter.coefficients_history.append(next_w_k)

        if verbose == True:

            if run == max_runs - 1 or print_every != -1 and run % print_every == 0:
                tac = time() - tic
                print('|error| = {:.02}\t Time: {:.03} ms'.format(
                    np.abs(error_k), (tac)*1000))
                tic = time()
        # tolerance break point

        if np.abs(error_k) < tolerance:
            if verbose == True:
                print(" ")
                print(" -- Ended at Run {} -- \n".format(run))
                print("Final |error| = {:.02}".format(np.abs(error_k)))
            break

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - total_tic)*1000))
    return {'outputs': outputs_vector,
            'errors': errors_vector, 'coefficients': Filter.coefficients_history}
