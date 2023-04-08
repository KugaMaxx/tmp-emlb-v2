import os
import cv2
import time
import numpy as np

from evtool.dvs import DvsFile
from evtool.utils import Player

from modules import kore
from modules import reclusive_event_denoisor as red
from modules import yang_noise as ynoise
from modules import double_window_filter as dwf
from modules import khodamoradi_noise as knoise
from modules import time_surface as ts
from modules import event_flow as evflow
from modules import multiLayer_perceptron_filter as mpf
from modules import event_denoise_convolution_network as edncnn

import matplotlib.pyplot as plt


# data = DvsFile.load('./data/demo/samples/demo-01.aedat4')
data = DvsFile.load('/home/kuga/Workspace/tmp-emlb/datasets/demo/samples/demo-01.aedat4')
# data['events'] = data['events'].shot_noise(data['size'], rate=5)
# idx = data['events'].hotpixel(data['size'], thres=1000)
# data['events'] = data['events'][idx]

# model = red.init(data['size'][0], data['size'][1])
# model = ynoise.init(data['size'][0], data['size'][1])
# model = dwf.init(data['size'][0], data['size'][1])
# model = knoise.init(data['size'][0], data['size'][1])
# model = ts.init(data['size'][0], data['size'][1])
# model = evflow.init(data['size'][0], data['size'][1])
# model = mpf.init(data['size'][0], data['size'][1])
model = edncnn.init(data['size'][0], data['size'][1])

# idx = model.run(data['events'])

# model_path = os.getcwd() + '/modules/_net/MLPF_2xMSEO1H20_linear_7.pt'
model_path = os.getcwd() + '/modules/_net/EDnCNN_all_trained_v9.pt'
idx = model.run(data['events'], model_path, threshold=0.5, batch_size=100)

# print(idx.sum(), data['events'].shape)
data['events'] = data['events'][idx]
for ts, ev in data['events'].slice('25ms'):
    img = ev.project(data['size'], 'monopolar')
    img[img > 0] = 255
    img = img.astype(np.uint8)
    
    cv2.namedWindow('result', cv2.WINDOW_FREERATIO) 
    cv2.imshow('result', img)
    cv2.waitKey(100)
