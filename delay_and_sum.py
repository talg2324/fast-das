import platform
import ctypes
import numpy as np
from scipy.interpolate import interp1d
import utils


class DAS():
    def __init__(self, use_gpu=False) -> None:

        # Load the shared library
        os_name = platform.system()

        if os_name == 'Windows' and use_gpu:
            self.lib = ctypes.CDLL('./bin/das_cu.dll')
            self.check_cuda()
        elif os_name == 'Windows' and not use_gpu:
            self.lib = ctypes.CDLL('./bin/libdas.dll')

        # Define the argument types for the C++ functions
        self.lib.envelope.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

        self.lib.delay_and_sum.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

        self.workspace = None

    def check_cuda(self):
        self.lib.cuda_valid.restype = None
        self.lib.cuda_valid.argtypes = [ctypes.POINTER(ctypes.c_bool)]
        valid = ctypes.c_bool()
        self.lib.cuda_valid(ctypes.byref(valid))

        if not valid:
            print("No NVIDIA GPU detected. Try updating CUDA toolkit. Reverting to CPU...")
            self.lib = ctypes.CDLL('./bin/das.dll')



    def load_vsx_workspace(self, filepath):
        self.workspace = utils.load_workspace(filepath)

    def init_delays(self, n_el, start_el=-1, transmission='pw'):

        if self.workspace is None:
            raise ValueError('No workspace loaded, call the load_vsx_workspace function first')
        elif start_el == -1:
            total_el = self.workspace['Trans']['ElementPos'].shape[0]
            start_el = total_el // 2 - n_el // 2

        im_res = (self.workspace['PData']['PDelta'][-1], self.workspace['PData']['PDelta'][0]) # x y z -> z x
        im_shape = tuple(self.workspace['PData']['Size'][:-1]) # z x
        
        pix_pos = np.zeros((im_shape[0], im_shape[1], 3))

        scan_origin = np.arange(im_res[1], im_shape[1] + im_res[1]) * im_res[1]
        scan_origin += self.workspace['PData']['Origin'][0]
        scan_origin = np.stack((scan_origin, 0*scan_origin, 0*scan_origin + self.workspace['PData']['Origin'][2]), -1)

        scan_dir = np.zeros((im_shape[1], 2))
        scan_depth = np.reshape(np.arange(im_res[0], im_shape[0]+im_res[0]) * im_res[0], (im_shape[0], 1))

        pix_pos[..., 0] = np.ones((im_shape[0], 1)) * scan_origin[:, 0] + scan_depth * np.sin(scan_dir[:, 0]) * np.cos(scan_dir[:, 1])
        pix_pos[..., 2] = np.ones((im_shape[0], 1)) * scan_origin[:, 2] + scan_depth * np.cos(scan_dir[:, 1])

        el_posx = self.workspace['Trans']['ElementPos'][start_el:start_el+n_el, 0] / self.workspace['Trans']['WavelenToMm']  # [wavelengths]
        el_posz = self.workspace['Trans']['ElementPos'][start_el:start_el+n_el, 2] / self.workspace['Trans']['WavelenToMm']  # [wavelengths]
        self.workspace['Trans']['lensCorrection'] /= self.workspace['Trans']['WavelenToMm']

        x = pix_pos[..., 0]  # [wavelengths]
        z = pix_pos[..., 2]  # [wavelengths]

        # Rx Focus
        # Rx delay is independent of transmission
        self.del_Rx = np.zeros((n_el, im_shape[0], im_shape[1]))

        Rx_const_delay = self.workspace['Trans']['lensCorrection'] - self.workspace['Receive']['StartDepth']

        for n in range(n_el):
            self.del_Rx[n, ...] = np.hypot(x-el_posx[n], z-el_posz[n])

        self.del_Rx = (self.del_Rx + Rx_const_delay) * self.workspace['Receive']['SamplesPerWave']

        # TX Focus
        Tx_const_delay = self.workspace['TW']['Peak'] + self.workspace['Trans']['lensCorrection'] - self.workspace['Receive']['StartDepth']

        if transmission == 'pw':
            self.gen_pw_delays(im_shape, el_posx, x, z)

        else:
            raise NotImplementedError('Transmission type not implemented- consider implementing yourself')
        
        self.del_Tx = (self.del_Tx + Tx_const_delay) * self.workspace['Receive']['SamplesPerWave']
        self.el_order = np.squeeze(self.workspace['Trans']['ConnectorES'][start_el:start_el+n_el]-1)
        self.n_el = n_el

    def gen_pw_delays(self, im_shape, el_posx, x, z):

        self.NA = self.workspace['NA']['NA_tot'][0, 0]
        self.del_Tx = np.zeros((self.NA, im_shape[0], im_shape[1]))
        for n in range(self.NA):
            phi = self.workspace['TX']['Steer'][n]
            el_delay = interp1d(el_posx, self.workspace['TX']['Delay'][n], kind='cubic', fill_value='extrapolate')

            self.del_Tx[n, ...] = z * np.cos(phi) + x * np.sin(phi)
            self.del_Tx[n, ...] += el_delay(0)  # Add delay to origin

    def beamform(self, rf_filepath):

        RF = utils.load_rf(rf_filepath)
        RF = RF[:, self.el_order].astype(np.float64).T

        start_sample = np.squeeze(self.workspace['Receive']['StartSample']).astype(np.int16) - 1
        end_sample = np.squeeze(self.workspace['Receive']['EndSample']).astype(np.int16)

        n_samples = end_sample[0] - start_sample[0]
        N = int(2**(np.ceil(np.log2(n_samples)))) # Round to next power of two

        envelope_real = np.zeros((self.NA, self.n_el, N), dtype=np.float64)
        envelope_imag = np.zeros_like(envelope_real)

        self.lib.envelope(
                            RF,
                            envelope_real,
                            envelope_imag,
                            start_sample,
                            end_sample,
                            self.NA,
                            self.n_el,
                            N,
                            RF.shape[1])

        print('Done calculating envelope')

        RF = envelope_real + 1j * envelope_imag

        na, height, width = self.del_Tx.shape
        
        us_im_real = np.zeros((na, height, width), dtype=np.float64)
        us_im_imag = np.zeros((na, height, width), dtype=np.float64)

        self.lib.delay_and_sum(
                                us_im_real,
                                us_im_imag,
                                envelope_real,
                                envelope_imag,
                                self.del_Tx,
                                self.del_Rx,
                                na,
                                self.n_el,
                                N,
                                height,
                                width)
        
        us_im = us_im_real + 1j * us_im_imag
        return np.abs(us_im)