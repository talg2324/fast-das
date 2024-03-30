import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from .das_lib import DASLib


class PyDASLib(DASLib):
    def __init__(self) -> None:
        logging.log(logging.WARNING, "DAS is slow in Python...")

    def envelope(self, RF: np.ndarray, na: int, n_el: int, N: int) -> np.ndarray:
        return hilbert(RF).transpose(1, 0, 2)

    def delay_and_sum(self, envelope, del_Tx, del_Rx):
        na, h, w = del_Tx.shape
        n_el = del_Rx.shape[0]

        beamformed = np.zeros((na, h, w), dtype=np.complex128)
        samples_axis = np.arange(0, envelope.shape[-1])
        for ang in range(na):
            for el in range(n_el):
                data_function = interp1d(samples_axis,
                                         envelope[ang, el, :],
                                         bounds_error=False, fill_value=0)
                beamformed[ang, ...] += data_function(del_Tx[ang, ...] + del_Rx[el, ...])

        return beamformed
