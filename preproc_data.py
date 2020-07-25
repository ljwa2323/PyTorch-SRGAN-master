import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    ROOT = os.path.abspath('.')
    hf_names = glob.glob(os.path.join(ROOT, 'raw data/hf/*.csv'))
    lf_names = glob.glob(os.path.join(ROOT, 'raw data/lf/*.csv'))

    hf_data = [np.asarray(pd.read_csv(hf_names[i], header=None)) for i in range(len(hf_names))]
    lf_data = [np.asarray(pd.read_csv(lf_names[i], header=None)) for i in range(len(lf_names))]

    hf_data = np.stack(hf_data, axis=2)
    lf_data = np.stack(lf_data, axis=2)

    hf_data = hf_data[:270*64*64, ...]
    hf_data = hf_data.reshape((-1, 64, 64, 3))
    hf_data = np.transpose(hf_data, (0, 3, 1, 2))

    lf_data = lf_data[:270*16*16, ...]
    lf_data = lf_data.reshape((-1, 16, 16, 3))
    lf_data = np.transpose(lf_data, (0, 3, 1, 2))

    np.savez(os.path.join(ROOT, 'datas/datas.npz'), hf=hf_data, lf=lf_data)

    # This is for check the data preprocessing result
    # for i in range(270):
    #     hf_tmp = hf_data[i, ...]
    #     lf_tmp = lf_data[i, ...]
    #     plt.subplot(1,2,1),plt.imshow(hf_tmp),\
    #     plt.subplot(1,2,2),plt.imshow(lf_tmp),\
    #     plt.savefig(os.path.join(ROOT,'figures confirm',str(i)+' hf vs lf.png'))
    #     plt.close()

    # plt.subplot(1,2,1),plt.imshow(hf_data[0,...])
    # plt.subplot(1,2,2),plt.imshow(lf_data[0,...]),plt.show()
    #
    # plt.imshow(hf_data[2,...]),plt.show()
    # plt.imshow(lf_data[2,...]),plt.show()




