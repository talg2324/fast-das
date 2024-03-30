import pkg_resources
import platform
import ctypes
import numpy as np

from .implementations import CalculateIn
from .das_lib import DASLib


class NativeDASLib(DASLib):
    def __init__(self, target) -> None:
        assert target.value[0] < 2
        self.load_lib(target)

        # Define the argument types for the C++ functions
        self.lib.envelope.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

        self.lib.delay_and_sum.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

    def load_lib(self, target: CalculateIn) -> None:
        # Load the shared library
        os_name = platform.system()

        lib_path = pkg_resources.resource_filename(
            'fastDAS', 'bin') + '/libdas'

        if target == CalculateIn.CUDA:
            lib_path += '_cu'

        if os_name == 'Windows':
            lib_path += '.dll'

        elif os_name == 'Linux':
            lib_path += '.so'

        else:
            raise NotImplementedError("fast-DAS is only pre-compiled for Linux and Windows. "
                                      "If you want, try using the scripted code in target=CalculateIn.Python")

        if target == CalculateIn.CUDA:
            self.check_cuda(lib_path)

        else:
            self.lib = ctypes.CDLL(lib_path)

    def check_cuda(self, lib_path: str) -> None:
        """
        checks if the CUDA environment is set up correctly

        Parameters
        ----------
        lib_path : str
            path to the shared library
        """

        lib = ctypes.CDLL(lib_path)

        lib.cuda_valid.restype = None
        lib.cuda_valid.argtypes = [ctypes.POINTER(ctypes.c_bool)]
        valid = ctypes.c_bool()
        lib.cuda_valid(ctypes.byref(valid))

        if not valid:
            raise EnvironmentError("No NVIDIA GPU detected. Try updating CUDA toolkit.")
        else:
            self.lib = lib

    def envelope(self, RF: np.ndarray, na: int, n_el: int, N: int) -> np.ndarray:
        envelope_real = np.zeros((na, n_el, N), dtype=np.float64)
        envelope_imag = np.zeros_like(envelope_real)
        start_sample = np.arange(na)*N
        samples_per_element = na * N

        self.lib.envelope(RF,
                        envelope_real,
                        envelope_imag,
                        start_sample,
                        na,
                        n_el,
                        N,
                        samples_per_element)
        return envelope_real + 1j * envelope_imag

    def delay_and_sum(self, envelope, del_Tx, del_Rx):
        na, h, w = del_Tx.shape
        _, n_el, N = envelope.shape

        us_im_real = np.zeros((na, h, w), dtype=np.float64)
        us_im_imag = np.zeros_like(us_im_real)

        self.lib.delay_and_sum(
            us_im_real,
            us_im_imag,
            np.ascontiguousarray(envelope.real),
            np.ascontiguousarray(envelope.imag),
            del_Tx,
            del_Rx,
            na,
            n_el,
            N, h, w)
        return us_im_real + 1j * us_im_imag
