import numpy as np
from functools import wraps

def ensure_real_signals(func):
    """
    Decorator to ensure that input signals (x and d) are real-valued.
    Raises TypeError if complex data is detected.
    """
    @wraps(func)
    def wrapper(self, x, d=None, *args, **kwargs):
        # Check primary input x
        if np.iscomplexobj(x):
            raise TypeError(f"{self.__class__.__name__} does not support complex inputs for 'x'.")
        
        # Check desired signal d (if provided)
        if d is not None and np.iscomplexobj(d):
            raise TypeError(f"{self.__class__.__name__} does not support complex inputs for 'd'.")
            
        return func(self, x, d, *args, **kwargs)
    return wrapper