import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot(path):
    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(26, 12), dpi=200)
    fig.set_facecolor('black')
    plt.axis('off')
    data = np.genfromtxt(path, delimiter=",")
    data = data[:, (0, 1, 7 ,8)]

    ids = np.unique(data[:, 1]).tolist()
    for idx, id in enumerate(ids):
        id_data = data[data[:, 1] == id]
        color = mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[15 * idx]]
        plt.plot(id_data[:, 2], id_data[:, 3], linewidth=5, color=color)

    name = osp.basename(path)[:-4]
    plt.savefig(f'plot_{name}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # path = '../../data/cache/mota_gt.txt'
    path = '../../data/cache/mota_pred.txt'
    plot(path)
