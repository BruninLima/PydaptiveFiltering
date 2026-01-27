import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any

class AdaptiveFilter(ABC):
    """
    Abstract Base Class for all Adaptive Filters in pydaptivefiltering.
    Based on the mathematical framework by Paulo S. R. Diniz.

    Attributes:
    -----------
    m : int
        Filter order (number of taps - 1).
    w : np.ndarray
        Current weight vector (coefficients).
    w_history : List[np.ndarray]
        Historical record of weight vectors during the adaptation process.
    supports_complex : bool
        Indicates if the filter supports complex-valued data.
    """

    supports_complex: bool = False 

    def __init__(self, m: int, w_init: Optional[Union[np.ndarray, list]] = None) -> None:
        """
        Initializes the adaptive filter.

        Parameters:
        -----------
        m : int
            Filter order.
        w_init : np.ndarray | list, optional
            Initial weights. Defaults to zeros if None.
        """
        self.m: int = m
        self.regressor: np.ndarray = np.zeros(self.m + 1, dtype=complex)

        if w_init is not None:
            self.w: np.ndarray = np.array(w_init, dtype=complex)
        else:
            self.w: np.ndarray = np.zeros(self.m + 1, dtype=complex)
            
        self.w_history: List[np.ndarray] = []
        self._record_history()

    def _record_history(self) -> None:
        """Internal method to store weight progression."""
        self.w_history.append(self.w.copy())

    def _validate_inputs(self, x, d):
        """Validação comum para todos os filtros adaptativos."""
        if x.size != d.size:
            raise ValueError(
                f"Tamanhos incompatíveis: input_signal ({x.size}) e "
                f"desired_signal ({d.size}) devem ter o mesmo comprimento."
            )
        
    def filter_signal(self, input_signal: Union[np.ndarray, list]) -> np.ndarray:
        """
        Processes the input signal using current filter coefficients.
        """
        x: np.ndarray = np.asarray(input_signal)
        n_samples: int = x.size
        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.m, dtype=complex)
        x_padded[self.m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.m + 1][::-1]
            y[k] = np.dot(self.w.conj(), x_k)

        return y

    @abstractmethod
    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        **kwargs: Any
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Abstract method for the adaptation algorithm.
        """
        pass

    def reset_filter(self, w_new: Optional[Union[np.ndarray, list]] = None) -> None:
        """Resets weights and history."""
        if w_new is not None:
            self.w = np.array(w_new, dtype=complex)
        else:
            self.w = np.zeros(self.m + 1, dtype=complex)
        self.w_history = []
        self._record_history()

def display_library_info() -> None:
    """Displays information about pydaptivefiltering chapters."""
    chapters: Dict[str, str] = {
        "3 & 4": "LMS Algorithms",
        "5": "RLS Algorithms",
        "6": "Set-Membership Algorithms",
        "7": "Lattice-based RLS Algorithms",
        "8": "Fast Transversal RLS Algorithms",
        "9": "QR Decomposition Based RLS Algorithms",
        "10": "IIR Adaptive Filters",
        "11": "Nonlinear Adaptive Filters",
        "12": "Subband Adaptive Filters",
        "13": "Blind Adaptive Filtering"
    }
    
    print("\n--- Pydaptive Filtering (Based on Paulo S. R. Diniz) ---")
    print(f"{'Chapter':<10} | {'Algorithm Area'}")
    print("-" * 50)
    for ch, area in chapters.items():
        print(f"Chapter {ch:<3} | {area}")
    print("-" * 50)

if __name__ == "__main__":
    display_library_info()