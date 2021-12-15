import os
import numpy as np

from tqdm import tqdm
from skimage import io
from vispy import app, gloo
from vispy.gloo import Program
from vispy.gloo.util import _screenshot
from vispy.util.transforms import perspective, translate, rotate
from moviepy.editor import VideoClip


DATA_ROOT = '../data/1A-DB/1A-2dtiff/'
SAVE_ROOT = '../result1/1A-DB/1A-2dtiff/'
INFO_NAME = 'C1_.info'


def read_data():
    """
    :return: imgs: [num * W * H]
    """
    data_root = DATA_ROOT
    # save_root = SAVE_ROOT
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)
    info_name = INFO_NAME

    f = open(data_root + info_name)

    # # # -------------------------------------------
    # img1 = io.imread(data_root + 'C1_542.tif')
    # img2 = read_tif(data_root + 'C1_306.tif')
    # img = (img1 + img2) / 2
    # io.imsave(data_root + 'C1_305.tif', img)
    # exit()
    # # # -------------------------------------------

    print('Loading images...')
    imgs = []
    for line in f:
        if '.tif' in line[1:11]:
            img_file = data_root + line[1:11]
            img = io.imread(img_file)
            # print(img_file, img.shape)
            # img[img >= 255] = 255
            # img = (img - img.min()) / (img.max() - img.min()) * 255
            # img = img.astype(np.uint8)
            # # io.imshow(img)
            # io.imsave(save_root + line[1:7] + '.jpg', img)
            img = img / 3092 * 255
            img = img.astype(np.uint8)
            imgs.append(img)

    imgs = np.array(imgs, dtype='uint8')
    np.save('../data/img_raw.npy', imgs)
    print('Images load completely!!! ')

    return imgs


def cube():
    """
    Build vertices for a color cube.

    :return:
    V  is the vertices
    """
    # imgs = read_data()
    imgs = np.load('../data/img_raw.npy')
    vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4)]

    d, w, h = imgs.shape
    print(d, w, h)
    print('Construct Vertices...')
    # Vertices colors
    c = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
         [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]

    # Vertices positions
    v = []
    pos_max = imgs < 168
    pos_min = imgs > 100
    index = np.argwhere(pos_max & pos_min)
    print(len(index))

    for i in tqdm(range(len(index))):
        print([index[i]])
        print(imgs[[index[i]]])
        v.append(((index[i] / [d, w, h] * 2 - 1.).tolist(), imgs[index[i]]/255 * c[3]))

    V = np.array(v, dtype=vtype)
    print('Vertices construct successfully')
    np.save('../data/data_256.npy', V)


if __name__ == '__main__':
    # read_data()
    cube()