import cv2
import numpy as np
import os


path = ''
savepath = ''
savename = 'sub'

filelist = os.listdir(path)
img = []
sub = 0
l = 8
for s in range(1):
    for i in range(l):
        T = cv2.imdecode(np.fromfile(path + '/' + filelist[8*s + i],dtype=np.uint8),-1)
        img.append(cv2.cvtColor(T, cv2.COLOR_BGR2GRAY).astype(np.uint16))
        sub += img[i]
    sub = sub/l
    cv2.imencode('.jpg', sub)[1].tofile(os.path.join(savepath, '%s.jpg') % savename)


