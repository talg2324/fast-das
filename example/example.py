import os
import time
import numpy as np
from matplotlib import pyplot as plt
from fastDAS import delay_and_sum as fd


def main():
    das = fd.DAS(use_gpu=True)
    das.load_vsx_workspace(os.environ['data_dir'] + 'Workspace.mat')
    das.init_delays(n_el=128, transmission='pw')

    t0 = time.time()
    us_im = das.beamform_file(os.environ['data_dir'] + 'RFData.mat')
    dt = time.time() - t0

    print('fastDAS: %.3f' % dt)

    n_ang = us_im.shape[0]
    for i in range(n_ang):
        plt.subplot(1, n_ang, i+1)
        plt.imshow(np.abs(us_im[i, ...]), cmap='gray')
        plt.axis('off')

        if i == n_ang//2:
            plt.title('fastDAS: %.3fsec' % dt)
        plt.xlabel('Acquisition #%d' %(i+1))
    plt.show()

if __name__ == "__main__":
    main()
