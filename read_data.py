import os
import numpy as np

from skimage import io
from tqdm import tqdm

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
    v = []
    num = 0
    vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4)]
    # c = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
    #      [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
    for line in f:
        if '.tif' in line[1:11]:
            # print(line[1:11])
            img_file = data_root + line[1:11]
            img = io.imread(img_file)
            img[img < 2e4] = 0
            index = np.argwhere(img > 3e4)

            for i in range(len(index)):
                pos = (np.hstack(([num], index[i])) / [d, w, h] * 2 - 1.).tolist()
                color = [0, img[index[i, 0], index[i, 1]]/65535, 0, 1]

                v.append((pos, color))
            num += 1

    print('Images load completely!!! ')
    V = np.array(v, dtype=vtype)
    np.save(SAVE_ROOT + 'data_C3.npy', V)
    print('Vertices construct successfully')


if __name__ == '__main__':
    read_data()