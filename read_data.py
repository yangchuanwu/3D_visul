import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from matplotlib.pyplot import plot,savefig

DATA_ROOT = 'G:/Data for huige-211122/1A-DB/1A-2dtiff/'
SAVE_ROOT = 'G:/Data for huige-211122/1A-DB/'
INFO_NAME = 'C3_.info'
d, w, h = 697, 2015, 2048


def read_data():
    """
    :return: imgs: [num * W * H]
    """
    data_root = DATA_ROOT
    save_root = SAVE_ROOT
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    info_name = INFO_NAME

    f = open(data_root + info_name)

    # # # -------------------------------------------
    # img1 = io.imread(data_root + 'C2_600.tif')
    # img2 = io.imread(data_root + 'C2_602.tif')
    # img = (img1 + img2) / 2
    # io.imsave(data_root + 'C2_601.tif', img)
    # exit()
    # # # -------------------------------------------

    print('Loading images...')
    x = range(65535)
    sum_num = np.zeros((65535,), dtype=np.int)
    for line in f:
        if len(line.split('"')) > 1:
            img_file = data_root + line.split('"')[1]
            print(img_file)
            img = io.imread(img_file)
            for i in range(len(sum_num)):
                sum_num[i] += np.sum(img == i)

    print('Images load completely!!! ')
    plt.plot(x, sum_num)
    plt.show()
    savefig('DB_C3.jpg')


if __name__ == '__main__':
    read_data()