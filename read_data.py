import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import io

DATA_ROOT = 'G:/Data for huige-211122'
DATA_NAME = 'DB'
INFO_NAME = 'C4'

if DATA_NAME == 'DB':
    d, w, h = 696, 2015, 2048
elif DATA_NAME == 'OB':
    d, w, h = 944, 2015, 2048
elif DATA_NAME == 'NCD':
    d, w, h = 1113, 2015, 2048
elif DATA_NAME == 'HFD':
    d, w, h = 1278, 2015, 2048
else:
    raise ValueError('Not ture data')

if INFO_NAME == 'C1':
    col = 1
    THRESHOLD = 280
elif INFO_NAME == 'C2':
    col = 2
    THRESHOLD = 800
elif INFO_NAME == 'C3':
    col = 1
    THRESHOLD = 2e4
elif INFO_NAME == 'C4':
    col = 0
    THRESHOLD = 2e4
else:
    raise ValueError('Not ture C')


def read_data():
    """
    :return: imgs: [num * W * H]
    """
    data_root = os.path.join(DATA_ROOT, DATA_NAME, '2dtiff/')
    info_name = INFO_NAME + '_.info'
    save_dir = 'imgs/' + DATA_NAME + '_' + INFO_NAME + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f = open(data_root + info_name)

    print('Loading images...')

    for line in f:
        if len(line.split('"')) > 1:
            img_file = data_root + line.split('"')[1]
            print(img_file)
            img = io.imread(img_file)
            # img[img > 1000] = 0

            label = np.zeros([w, h, 3])
            if (img > THRESHOLD).any():
                img1 = img.copy()
                img1[img1 < THRESHOLD] = 0
                label[:, :, col] = img1.copy() / 1000 * 255
            img = img/img.max() * 255
            img = img.astype(np.uint8)
            label = label.astype(np.uint8)
            io.imsave(save_dir + line.split('"')[1][:-4] + '.jpg', img)
            io.imsave(save_dir + line.split('"')[1][:-4] + '_l.png', label)


def count_pixel():
    data_root = os.path.join(DATA_ROOT, DATA_NAME, '2dtiff/')
    info_name = INFO_NAME + '_.info'

    f = open(data_root + info_name)

    print('Loading images...')
    x = range(65536)
    sum_num = np.zeros((65536,), dtype=np.int)

    for line in f:
        if len(line.split('"')) > 1:
            img_file = data_root + line.split('"')[1]
            print(img_file)
            img = io.imread(img_file)

            uniq, count = np.unique(img, return_counts=True)
            for i in range(len(uniq)):
                sum_num[uniq[i]] += count[i]

    print('Images load completely!!! ')
    plt.plot(x, sum_num)
    plt.savefig('data_analysis/' + DATA_NAME + '_' + INFO_NAME + '.jpg')
    np.save('data_analysis/' + DATA_NAME + '_' + INFO_NAME + '.npy', [x, sum_num])


def plot_bar():
    data_root = 'data_analysis/' + DATA_NAME + '_' + INFO_NAME + '.npy'
    save_root = os.path.join('fig', DATA_NAME + '_' + INFO_NAME)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data = np.load(data_root)
    x = data[0]
    sum_num = data[1]
    # thr = [1e7, 1e6, 1e5, 5e4, 1e4, 8e3, 6e3, 4e3, 2e3, 1e3, 900, 800, 700, 600, 500, 400, 300, 200, 100]

    for i in tqdm(range(0, 1000, 50)):
        x1 = x[i:50+i]
        sum_num1 = sum_num[i:50+i]

        plt.plot(x1, sum_num1)
        plt.xlabel('像素值')
        plt.ylabel('出现次数')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.savefig('data_analysis/' + DATA_NAME + '_' + INFO_NAME + '.jpg')
        plt.savefig(save_root + '/{}.jpg'.format(i))
        plt.cla()


if __name__ == '__main__':
    read_data()
    # plot_bar()