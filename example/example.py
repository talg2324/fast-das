import os
import numpy as np
from matplotlib import pyplot as plt

from vsx import from_verasonics
import fastDAS as fd


def main():
    RF, del_Tx, del_Rx = from_verasonics(os.environ['data_dir'])
    das = fd.DAS(target=fd.CalculateIn.CUDA)
    us_im = das.beamform(RF, del_Tx, del_Rx)

    n_ang = us_im.shape[0]
    for i in range(n_ang):
        plt.subplot(1, n_ang, i+1)
        plt.imshow(np.abs(us_im[i, ...]), cmap='gray')
        plt.axis('off')
        plt.xlabel('Acquisition #%d' %(i+1))
    plt.show()


if __name__ == "__main__":
    main()
