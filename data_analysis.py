import os
import numpy as np

from skimage import io

DATA_ROOT = 'G:/Data for huige-211122'
DATA_NAME = 'DB'
INFO_NAME = 'C1'
d, w, h = 696, 2015, 2048
THRESHOLD = 280


def read_data():
    """
    :return: imgs: [num * W * H]
    """
    data_root = os.path.join(DATA_ROOT, DATA_NAME, '2dtiff/')
    info_name = INFO_NAME + '_.info'

    f = open(data_root + info_name)

    print('Loading images...')
    v = []
    num = 0
    vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4)]
    c = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
         [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
    if INFO_NAME == 'C1':
        color = c[0]
    elif INFO_NAME == 'C2':
        color = c[1]
    elif INFO_NAME == 'C3':
        color = c[3]
    elif INFO_NAME == 'C4':
        color = c[7]
    else:
        raise ValueError('Not ture C')

    for line in f:
        if len(line.split('"')) > 1:
            img_file = data_root + line.split('"')[1]
            img = io.imread(img_file)
            img[img > 1000] = 0

            index = np.argwhere(img > THRESHOLD)
            for i in range(len(index)):
                pos = (np.hstack(([num], index[i])) / [d, w, h] * 2 - 1.).tolist()
                vcolor = img[index[i, 0], index[i, 1]]/1000 * np.array(color)
                vcolor[3] = 1

                v.append((pos, vcolor))
            num += 1
            print(num)

    print('Images load completely!!! ')
    V = np.array(v, dtype=vtype)
    np.save('data_V/' + DATA_NAME + '_' + INFO_NAME + '.npy', V)
    print('Vertices construct successfully')


if __name__ == '__main__':
    read_data()
