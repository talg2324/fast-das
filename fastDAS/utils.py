import numpy as np


def pad_to_nearest_power_of_2(RF: np.ndarray, n_samples: int) -> np.ndarray:
    N = int(2**np.ceil(np.log2(n_samples)))
    if N > n_samples:
        pad_0 = (0, 0)
        pad_N = (0, N-n_samples)
        RF = np.pad(RF, (pad_0, pad_0, pad_N))
    return RF, N
