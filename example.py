import os
import time
from matplotlib import pyplot as plt
from delay_and_sum import DAS


def main():
    das = DAS(use_gpu=False)
    das.load_vsx_workspace(os.environ['data_dir'] + 'Workspace.mat')
    das.init_delays(n_el=128, transmission='pw')

    t0 = time.time()
    us_im = das.beamform(os.environ['data_dir'] + 'RFData.mat')
    dt = time.time() - t0

    print('CDAS: %.3f' % dt)

    # t0 = time.time()
    # us_im2 = tmp_test(das)
    # dt2 = time.time() - t0

    # print('PyDAS: %.3f' % dt2)

    n_ang = us_im.shape[0]
    for i in range(n_ang):
        plt.subplot(2, n_ang, i+1)
        plt.imshow(us_im[i, ...], cmap='gray')
        plt.axis('off')

        if i == n_ang//2:
            plt.title('CDAS: %.3f' % dt)

        # plt.subplot(2, n_ang, i+1 + n_ang)
        # plt.imshow(us_im2[i, ...], cmap='gray')
        # plt.axis('off')

        # if i == n_ang//2:
        #     plt.title('PyDAS: %.3f' % dt2)
    plt.show()


def tmp_test(das):

    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.signal import hilbert
    import utils

    RF_tot = utils.load_rf(os.environ['data_dir'] + 'RFData.mat')
    RF_tot = RF_tot[:, das.el_order].astype(np.float64).T

    start_sample = np.squeeze(das.workspace['Receive']['StartSample']).astype(np.int16) - 1
    end_sample = np.squeeze(das.workspace['Receive']['EndSample']).astype(np.int16)

    n_samples = end_sample[0] - start_sample[0]
    RF = np.zeros((n_samples, das.NA, das.n_el), dtype=np.complex128)

    for n in range(das.NA):
        start_samp = das.workspace['Receive']['StartSample'][n, 0] - 1
        end_samp = das.workspace['Receive']['EndSample'][n, 0]
        RF[:n_samples, n, :] = hilbert(np.squeeze(RF_tot[:, start_samp:end_samp]).T)

    # DAS
    R_n = np.arange(0.5, RF.shape[0]+0.5)

    out = []

    for nTx in range(das.NA):
        Focused_RF = np.zeros((das.workspace['PData']['Size'][0], das.workspace['PData']['Size'][1]), dtype=np.complex128)
        for nRx in range(128):
            DAS = interp1d(R_n, RF[:, nTx, nRx], bounds_error=False, fill_value=0)
            Focused_RF += DAS(das.del_Rx[nRx, ...] + das.del_Tx[nTx, ...])  # * TXPD
        out.append(Focused_RF.copy())

    return np.abs(out)


if __name__ == "__main__":
    main()
