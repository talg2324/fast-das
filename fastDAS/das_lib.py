from abc import ABC, abstractmethod
import numpy as np


class DASLib(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def envelope(self, RF: np.ndarray, na: int, n_el: int, N: int) -> np.ndarray:
        pass

    @abstractmethod
    def delay_and_sum(self):
        pass
