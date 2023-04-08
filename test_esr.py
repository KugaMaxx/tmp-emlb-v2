import os
import cv2
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evtool.dvs import DvsFile
from evtool.utils._player.mpl_player import Animator


def search_file(file, path='./results'):
    search_path = f"{path}"
    search_name = f"{file}.pkl"

    result = []
    for root, dirs, files in os.walk(f"{path}"):
        for name in files:
            if name == f"{file}.pkl":
                result.append(os.path.join(root, name))

    return result

def calc_esr(event, size=(260, 346)):
    cnt = event.project(size, 'monopolar')
    N = len(event)
    K = size[0] * size[1]

    alpha = (K - (0.5**cnt).sum()) / K
    return 1000 * np.sqrt((cnt * cnt).sum() * alpha / (N * N))


# 读文件
file_name = 'LEGO-ND16-2'
data_base = dict()
for f in search_file(file_name, '/media/kuga/瓜果山/results/final/'):
    print(f"find {f.split('/')[-4]} ({f})")
    data_base[f.split('/')[-4]] = DvsFile.load(f)

# 集体切片
interval, size = '25ms', (260, 346)
from_timestamp = 0
packet, packet_len = dict(), []
for i, (key, data) in enumerate(data_base.items()):
    print(f"slice {key}")
    if i == 0:
        from_timestamp = data['events'].timestamp[0]
    packet[key] = [ev for ev in data['events'].slice(interval, from_timestamp)]
    packet_len.append(len(packet[key]))
packet_len = min(packet_len)

# 计算ESR
fig, axs = plt.subplots(1, len(data_base), figsize=(12, 3))
# for i in range(1, packet_len):
#     for j, (key, pack) in enumerate(packet.items()):
#         timestamp, event = pack[i]
#         calc_esr(event, size)

obj = []
for j, (key, pack) in enumerate(packet.items()):
    obj.append(axs[j].imshow(np.zeros(size),
                                vmin=-1, vmax=1, cmap=plt.set_cmap('bwr')))
    axs[j].set_axis_off()


def update(i):
    for j, (key, pack) in enumerate(packet.items()):
        timestamp, event = pack[i]
        obj[j].set_data(event.project(size))
        axs[j].set_title(key + f' esr:{calc_esr(event, size):.5f}' + '\n' + f' cnt:{len(event)}')

Animator(fig, update, ticks=[i for i in range(packet_len)])
plt.show()
