import numpy as np
import cv2
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
# path = 'E:/YangHR/oganoid/data/ExternalData/cellos_data/cellos_data/'
# pathsave = 'C:/Users/Dell/Desktop/zancun/cello'
# # data = np.load(path1 +path2)
# file_name_list = os.listdir(path)
# # for i in range(len(file_name_list)):
# for i in range(500):
#     image = np.zeros((1080,1080,3)).astype(np.uint8)
#     for j in range(3):
#         fileName = os.path.splitext(file_name_list[3*i + j])[0]
#         imgpath = path + fileName + '.tiff'
#         img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#         img = (img - img.min()) / (img.max() - img.min()) * 255
#         img = img.astype(np.uint8)
#         image[:, : ,j] = img
#     cv2.imencode('.tiff', image)[1].tofile(os.path.join(pathsave, '%s.tiff') % fileName)


path = '/home/hryang/organoid/Well C003/'
pathsave = '/home/hryang/organoid/result/'
file_name_list = os.listdir(path)
image = np.zeros((512,672,17)).astype(np.uint8)
for i in range(int(len(file_name_list)/2)):
# for i in range(500):
    fileName = os.path.splitext(file_name_list[i])[0]
    imgpath = path + fileName + '.tif'
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    image[:,:,i] = img
plt.imshow(image, cmap='gray')
plt.show()
# cv2.imencode('.tiff', image)[1].tofile(os.path.join(pathsave, '%s.tiff') % fileName)