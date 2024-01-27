import numpy as np
from scipy.interpolate import interp1d
import json
import os

ROOTDIR = os.path.split(__file__)[0]

with open(os.path.join(ROOTDIR, 'vtm.json'), 'r') as f:
    data = json.load(f)

bpp_psnr = interp1d(data['results']['bpp'], data['results']['psnr-rgb'])
bpp_time = interp1d(data['results']['bpp'], data['results']['decoding_time'])

print(bpp_psnr(0.9937)-0.2, bpp_time(0.9937)*3)