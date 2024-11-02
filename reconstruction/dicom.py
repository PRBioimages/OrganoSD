import pydicom
from matplotlib import pyplot as plt
import os
import numpy as np

path = '/home/hryang/CTC/dicomset/MRIm001.dcm'
img_upload_path = '/home/hryang/CTC/dicomresult/MRIm001.jpg'
data = pydicom.dcmread(path)
if len(data.pixel_array.shape) > 2:
    for index in range(int(data.pixel_array.shape[0])):
        img = np.asarray(data.pixel_array[index], dtype='uint16')
        plt.imsave(img_upload_path, img, cmap=plt.cm.bone)
else:
    img = np.asarray(data.pixel_array, dtype='uint16')
    # img = (img - img.min()) / (img.max() - img.min()) * 255
    plt.imsave(img_upload_path, img, cmap = plt.cm.bone)

