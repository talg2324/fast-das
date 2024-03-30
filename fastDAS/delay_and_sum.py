import numpy as np
from .utils import pad_to_nearest_power_of_2
from .implementations import CalculateIn
from .py_lib import PyDASLib
from .native_lib import NativeDASLib


class DAS():
    def __init__(self, target: CalculateIn = CalculateIn.CPP) -> None:
        """
        delay-and-sum object
        this class is a high-level link to the underlying libraries executing the calculations

        Parameters
        ----------
        target : CalculateIn, optional
            target language implementatio, either cpp, cuda, or python,
            passed as an enumerable. by default CalculateIn.CPP
        """
        if target == CalculateIn.PYTHON:
            self.lib = PyDASLib()
        else:
            self.lib = NativeDASLib(target)

    def beamform(self, RF: np.ndarray, del_Tx: np.ndarray, del_Rx: np.ndarray) -> np.ndarray:
        """
        utility function combining the envelope and delay-and-sum steps
        see descriptions of those functions for more details
        """
        env = self.envelope(RF)
        return self.delay_and_sum(env, del_Tx, del_Rx)

    def envelope(self, RF: np.ndarray) -> np.ndarray:
        """
        calculate the signal envelope along the temporal axis using the hilbert transform

        Parameters
        ----------
        RF : np.ndarray
            raw RF data of shape [n_el, n_acq, n_samples]

        Returns
        -------
        np.ndarray
            complex hilbert transform of the input
            NOTE: the return value has shape [n_acq, n_el, N]
            this transposes the first two dimensions and pads n_samples to N (the next power of two)
        """
        n_el, na, n_samples = RF.shape
        RF, N = pad_to_nearest_power_of_2(RF, n_samples)
        return self.lib.envelope(RF.astype(np.float64), na, n_el, N)

    def delay_and_sum(self, envelope: np.ndarray, del_Tx: np.ndarray, del_Rx: np.ndarray) -> np.ndarray:
        """
        delay and sum algorithm implemented using linear interpolation

        Parameters
        ----------
        envelope : np.ndarray
            complex hilbert transform of the signal
            should have shape [n_acq, n_el, N]
        del_Tx : np.ndarray
            transmit delay matrix for an output of shape HxW
            should have shape [n_acq, H, W]
        del_Rx : np.ndarray
            receive delay matrix for an output of shape HxW
            should have shape [n_el, H, W]

        Returns
        -------
        np.ndarray
            complex beamformed ultrasound image beamformed for each transmission individually
            will have the same shape as the del_Tx input
        """
        return self.lib.delay_and_sum(envelope, del_Tx, del_Rx)
