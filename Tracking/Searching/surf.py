import cv2
import numpy as np

def sift_kp(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(600)
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None)
    return kp_image, kp, des

def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    #对两个匹配选择最佳的匹配
    matches = bf.knnMatch(des1, des2, k=2)
    # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good



path1 = ''
path2 = ''
img1 = cv2.imdecode(np.fromfile(path1,dtype=np.uint8),-1)
img2 = cv2.imdecode(np.fromfile(path2,dtype=np.uint8),-1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


kpimg1, kp1, des1 = sift_kp(img1)
kpimg2, kp2, des2 = sift_kp(img2)
print('descriptor1:', des1.shape, 'descriptor2', des2.shape)

goodMatch = get_good_match(des1, des2)
all_goodmatch_img= cv2.drawMatches(img1, kp1, img2, kp2, goodMatch, None, flags=2)
goodmatch_img = cv2.drawMatches(img1, kp1, img2, kp2, goodMatch[:50], None, flags=2)



cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
cv2.namedWindow('all_goodmatch_img',cv2.WINDOW_NORMAL)
cv2.namedWindow('goodmatch_img',cv2.WINDOW_NORMAL)

cv2.imshow('img1',np.hstack((img1,kpimg1)))
cv2.imshow('img2',np.hstack((img2,kpimg2)))
cv2.imshow('all_goodmatch_img', all_goodmatch_img)
cv2.imshow('goodmatch_img', goodmatch_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# kp1, des1 = sift.compute(img1, kp1)