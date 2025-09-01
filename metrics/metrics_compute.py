import cv2
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import multiprocessing
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import tqdm
import pywt
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser( description='The 7 handcrafted features of organoid are calculated')
parser.add_argument('--image_root', default='/home/hryang/Detecseg/sig_metric/nasal_img/',
                    type=str, help='存储单类器官图像的文件夹目录')
parser.add_argument('--mask_root', default='/home/hryang/Detecseg/sig_metric/nasal_mask/',
                    type=str, help='与image_root对应的mask目录')
parser.add_argument('--metrics_savepath', default='/home/hryang/Detecseg/sig_metric/nasal.csv',
                    type=str, help='计算结果的存储路径')
parser.add_argument('--msei_scales', default=[1, 3, 5, 7],
                    type=list, help='尺度参数')
parser.add_argument('--msei_wlevel', default=1,
                    type=int, help='小波分解的层级')
parser.add_argument('--locoefa_nummode', default=20,
                    type=int, help='椭圆模式数目，值越大，拟合越准确，但会导致计算量增大')
parser.add_argument('--DEBUG', default=False,
                    type=bool, help='调参(locoefa_nummode)')
args = parser.parse_args()


def calculate_perimeter(contour, is_closed=True):
    """
    自定义计算轮廓周长

    参数:
        contour (ndarray): 输入的轮廓数据，形状为 (N, 1, 2)，每个点包含 x 和 y 坐标
        is_closed (bool): 是否闭合轮廓，默认为 True

    返回:
        float: 计算出的轮廓周长
    """
    # 将轮廓转化为一维数组，便于操作
    contour = contour.reshape(-1, 2)  # 转为 (N, 2) 形式
    perimeter = 0.0
    # 计算相邻点之间的距离
    for i in range(len(contour) - 1):
        x1, y1 = contour[i]
        x2, y2 = contour[i + 1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        perimeter += distance
    # 如果是闭合轮廓，还需计算最后一个点与第一个点的距离
    if is_closed:
        x1, y1 = contour[-1]
        x2, y2 = contour[0]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        perimeter += distance
    return perimeter

def epi_calculate(contour, area):
    # perimeter_actual = cv2.arcLength(contour, True)
    perimeter_actual = calculate_perimeter(contour, is_closed=True)
    perimeter_equal = np.sqrt(4 * np.pi * area)
    epi = perimeter_actual / perimeter_equal - 1
    return epi

def sample_entropy(signal, m=2, r=0.2):
    '''
    通过KDTree优化距离计算样本熵，msei的中间步骤
    signal: 输入时间序列
    m: 嵌入维度
    r: 容差比例
    '''
    N = len(signal)
    # 将信号转换为m维和m+1维向量
    X_m = np.array([signal[i:i + m] for i in range(N - m)])
    X_m1 = np.array([signal[i:i + m + 1] for i in range(N - m - 1)])
    # 使用KDTree来加速相似度计算
    tree = KDTree(X_m)
    tree_m1 = KDTree(X_m1)
    def count_similar_vectors(X, tree, r):
        """
        计算给定KDTree中，每个向量在r容差内的相似向量比例
        """
        count = 0
        for i in range(len(X)):
            # 查询与第i个向量在给定半径r内的所有向量
            dist = tree.query_radius(X[i:i+1], r)
            count += len(dist[0])  # 计算符合条件的向量数目
        return count / len(X)
    phi_m = count_similar_vectors(X_m, tree, r)  # m维向量的相似度
    phi_m1 = count_similar_vectors(X_m1, tree_m1, r)  # m+1维向量的相似度
    if phi_m1 == 0:
        return np.inf
    return -np.log(phi_m1 / phi_m)

def msei_calculate(contour, scales=[1, 2, 3, 4, 5], wave_level=4):
    '''
    参考文献：Automated evaluation of tumor spheroid behavior in 3D culture using deep learning-based recognition
    '''
    #构造输入信号:用每个轮廓点与质心的距离作为初始信号，然后利用小波变换滤除其低频分量
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    signal_ori = np.sqrt((contour[:, 0, 0] - cx) ** 2 + (contour[:, 0, 1] - cy) ** 2)
    coeffs = pywt.wavedec(signal_ori, 'haar', level=wave_level)
    signal = pywt.waverec([None] + coeffs[1:], 'haar')

    mse_values = []
    for scale in scales:
        # 在当前尺度下重采样信号,再计算样本熵
        resampled_signal = [np.mean(signal[i:i + scale]) for i in range(0, len(signal), scale)]
        mse_values.append(sample_entropy(resampled_signal, 2, 0.2))
    return sum(mse_values) / len(mse_values)

def solid_calculate(contour, area):
    # 计算凸包并求面积
    convex_hull = cv2.convexHull(contour)
    convex_hull_area = cv2.contourArea(convex_hull)
    return area / convex_hull_area if convex_hull_area != 0 else 0

def eccentricity_calculate(contour):
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1])  # 长轴
    minor_axis = min(ellipse[1])  # 短轴
    e = np.sqrt(1 - minor_axis ** 2 / major_axis ** 2)
    return e

def transmittance_calculate(mask, image):
    #计算背景像素值
    mask_zero_positions = np.where(mask == 0)
    zero_positions = list(zip(mask_zero_positions[0], mask_zero_positions[1]))
    n = 40
    if len(zero_positions) < n:
        print(f"掩膜中有效的点数不足 {n} 个，仅选择 {len(zero_positions)} 个点")
        n = len(zero_positions)
    selected_points = np.random.choice(len(zero_positions), size=n, replace=False)
    selected_pixel_values = [image[zero_positions[i]] for i in selected_points]
    background_value = np.mean(selected_pixel_values)

    mask = mask // 255
    image = abs(image - background_value) * mask
    average_gray = np.sum(image) / np.sum(mask)
    transmittance = 1 - average_gray / 255
    return transmittance


def locoefa_calculate(contour, N_modes=50, DEBUG=True):
    '''
        参考文献：Morphometrics of complex cell shapes: lobe contribution elliptic Fourier analysis (LOCO-EFA)
    '''
    contour, mode = read_example_data(pd.DataFrame(contour.squeeze(axis=1), columns=['x', 'y']), N_modes=N_modes)
    mode, contour = EFA(contour, mode, DEBUG=DEBUG)
    mode = LOCOEFA(mode, DEBUG=DEBUG)

    ln_ = np.array([mode.locoL[i] for i in range(len(mode))])
    hist, bin_edges = np.histogram(ln_, bins=1000)
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    #调参，根据重建出来的形状确定使用多少个mode
    if (DEBUG):
        ### compute and make plots
        tp = np.linspace(0, 1, 100)
        fignum = np.ceil(np.sqrt(len(mode))).astype(int)
        fig, ax = plt.subplots(fignum, fignum, figsize=(28, 28))
        ax = ax.flatten()
        for mm in tqdm.trange(len(mode), desc='Max_modes'):
            x_loco, y_loco = reconstruct_contour(mode, tp=tp, rec_type='LOCOEFA', first_mode=0,
                                                             last_mode=mm)
            x_efa, y_efa = reconstruct_contour(mode, tp=tp, rec_type='EFA', first_mode=0, last_mode=mm)
            ax[mm].plot(contour.x, contour.y, '-b')
            ax[mm].plot(x_loco, y_loco, '-r')
            ax[mm].plot(x_efa, y_efa, '-', color='orange')
            ax[mm].set_title('Mode %d' % mm, fontsize=8)
            ax[mm].set_xticks([])
            ax[mm].set_yticks([])
        plt.show()

    mode = mode.values.flatten().tolist()
    return mode, ln_, entropy

def EFA(contour, mode, DEBUG=False):
    if (DEBUG):
        print('Computing EFA coefficients...')
    N_points = len(contour.x)
    N_modes = len(mode.alpha) - 2
    # Eq. 5: DeltaX, DeltaY, DeltaT, T
    contour.deltax[1:] = np.diff(contour.x)
    contour.deltay[1:] = np.diff(contour.y)
    contour.deltat[1:] = np.sqrt(contour.deltax[1:] ** 2 + contour.deltay[1:] ** 2)
    contour.t = np.cumsum(contour.deltat)
    T = contour.t.values[-1]
    # extract info as numpy arrays for fast computations
    deltax = contour.deltax.values
    deltay = contour.deltay.values
    deltat = contour.deltat.values
    t = contour.t.values
    xi = contour.xi.values
    epsilon = contour.epsilon.values
    # Eq. 7: : sumDeltaxj, sumDeltayj, xi, epsilon
    for i in range(2, N_points):
        contour.sumdeltaxj[i] = contour.sumdeltaxj[i - 1] + contour.deltax[i - 1]
        contour.sumdeltayj[i] = contour.sumdeltayj[i - 1] + contour.deltay[i - 1]
        contour.xi[i] = contour.sumdeltaxj[i] - contour.deltax[i] / contour.deltat[i] * contour.t[i - 1]
        contour.epsilon[i] = contour.sumdeltayj[i] - contour.deltay[i] / contour.deltat[i] * contour.t[i - 1]

    # Equation 7: alpha0, gamma0
    mode.alpha[0] = contour.x[0]
    mode.gamma[0] = contour.y[0]
    mode.alpha[0] += np.sum(
        (deltax[1:] / (2. * deltat[1:]) * (t[1:] ** 2 - t[:-1] ** 2) + xi[1:] * (t[1:] - t[:-1])) / T)
    mode.gamma[0] += np.sum(
        (deltay[1:] / (2. * deltat[1:]) * (t[1:] ** 2 - t[:-1] ** 2) + epsilon[1:] * (t[1:] - t[:-1])) / T)
    for j in range(1, N_modes):
        mode.alpha[j] = np.sum(
            deltax[1:] / deltat[1:] * (np.cos(2. * j * np.pi * t[1:] / T) - np.cos(2 * j * np.pi * t[:-1] / T))) * T / (
                                    2. * j ** 2 * np.pi ** 2)
        mode.beta[j] = np.sum(
            deltax[1:] / deltat[1:] * (np.sin(2. * j * np.pi * t[1:] / T) - np.sin(2 * j * np.pi * t[:-1] / T))) * T / (
                                   2. * j ** 2 * np.pi ** 2)
        mode.gamma[j] = np.sum(
            deltay[1:] / deltat[1:] * (np.cos(2. * j * np.pi * t[1:] / T) - np.cos(2 * j * np.pi * t[:-1] / T))) * T / (
                                    2. * j ** 2 * np.pi ** 2)
        mode.delta[j] = np.sum(
            deltay[1:] / deltat[1:] * (np.sin(2. * j * np.pi * t[1:] / T) - np.sin(2 * j * np.pi * t[:-1] / T))) * T / (
                                    2. * j ** 2 * np.pi ** 2)
    if (DEBUG):
        print("\n\nEFA coefficients:\n=================\n\n")
        for j in range(N_modes):
            print("mode %d:\n" % (j))
            print("(%f\t%f)\n" % (mode.alpha[j], mode.beta[j]))
            print("(%f\t%f)\n\n" % (mode.gamma[j], mode.delta[j]))
    return mode, contour

def LOCOEFA(mode, DEBUG=False):
    if (DEBUG):
        print('Computing LOCO-EFA coefficients...')
    N_modes = len(mode.alpha) - 2

    # Equation 14: tau1
    mode.tau[1] = 0.5 * np.arctan2(2. * (mode.alpha[1] * mode.beta[1] + mode.gamma[1] * mode.delta[1]),
                                   mode.alpha[1] ** 2 + mode.gamma[1] ** 2 - mode.beta[1] ** 2 - mode.delta[1] ** 2)

    # Below eq. 15: alpha1prime, gamma1prime
    mode.alphaprime[1] = mode.alpha[1] * np.cos(mode.tau[1]) + mode.beta[1] * np.sin(mode.tau[1])
    mode.gammaprime[1] = mode.gamma[1] * np.cos(mode.tau[1]) + mode.delta[1] * np.sin(mode.tau[1])

    # Equation 16: rho
    mode.rho[1] = np.arctan2(mode.gammaprime[1], mode.alphaprime[1])

    # Equation 17: tau1
    if (mode.rho[1] < 0.):
        mode.tau[1] += np.pi

    # Equation 18: alphastar, betastar, gammastar, deltastar
    mode.alphastar[1:N_modes + 1] = mode.alpha[1:N_modes + 1] * np.cos(
        np.arange(1, N_modes + 1) * mode.tau[1]) + mode.beta[1:N_modes + 1] * np.sin(
        np.arange(1, N_modes + 1) * mode.tau[1])
    mode.betastar[1:N_modes + 1] = -mode.alpha[1:N_modes + 1] * np.sin(
        np.arange(1, N_modes + 1) * mode.tau[1]) + mode.beta[1:N_modes + 1] * np.cos(
        np.arange(1, N_modes + 1) * mode.tau[1])
    mode.gammastar[1:N_modes + 1] = mode.gamma[1:N_modes + 1] * np.cos(
        np.arange(1, N_modes + 1) * mode.tau[1]) + mode.delta[1:N_modes + 1] * np.sin(
        np.arange(1, N_modes + 1) * mode.tau[1])
    mode.deltastar[1:N_modes + 1] = -mode.gamma[1:N_modes + 1] * np.sin(
        np.arange(1, N_modes + 1) * mode.tau[1]) + mode.delta[1:N_modes + 1] * np.cos(
        np.arange(1, N_modes + 1) * mode.tau[1])

    # Equation 9: r
    mode.r[1] = mode.alphastar[1] * mode.deltastar[1] - mode.betastar[1] * mode.gammastar[1]

    # Equation 19: betastar, deltastar
    if (mode.r[1] < 0.):
        mode.betastar[1:N_modes + 1] = -mode.betastar[1:N_modes + 1]
        mode.deltastar[1:N_modes + 1] = -mode.deltastar[1:N_modes + 1]

    # Equation 20: a, b, c, d
    mode.a[0] = mode.alpha[0]
    mode.c[0] = mode.gamma[0]

    mode.a[1:N_modes + 1] = mode.alphastar[1:N_modes + 1]
    mode.b[1:N_modes + 1] = mode.betastar[1:N_modes + 1]
    mode.c[1:N_modes + 1] = mode.gammastar[1:N_modes + 1]
    mode.d[1:N_modes + 1] = mode.deltastar[1:N_modes + 1]

    if (DEBUG):
        print("\n\nmodified EFA coefficients:\n==========================\n\n")
        for i in range(N_modes):
            print("mode %d:\n" % i)
            print("(%f\t%f)\n" % (mode.a[i], mode.b[i]))
            print("(%f\t%f)\n\n" % (mode.c[i], mode.d[i]))

    if (DEBUG):
        print("\n\nLambda matrices:\n================\n\n")

    ## this can all be optimized, but probably not worth it
    for i in range(1, N_modes + 1):
        # Equation 26: phi
        mode.phi[i] = 0.5 * np.arctan2(2. * (mode.a[i] * mode.b[i] + mode.c[i] * mode.d[i]),
                                       mode.a[i] ** 2 + mode.c[i] ** 2 - mode.b[i] ** 2 - mode.d[i] ** 2)

        # Below eq. 27: aprime, bprime, cprime, dprime
        mode.aprime[i] = mode.a[i] * np.cos(mode.phi[i]) + mode.b[i] * np.sin(mode.phi[i])
        mode.bprime[i] = -mode.a[i] * np.sin(mode.phi[i]) + mode.b[i] * np.cos(mode.phi[i])
        mode.cprime[i] = mode.c[i] * np.cos(mode.phi[i]) + mode.d[i] * np.sin(mode.phi[i])
        mode.dprime[i] = -mode.c[i] * np.sin(mode.phi[i]) + mode.d[i] * np.cos(mode.phi[i])

        # Equation 27: theta
        mode.theta[i] = np.arctan2(mode.cprime[i], mode.aprime[i])

        # Equation 25: Lambda
        mode.lambda1[i] = np.cos(mode.theta[i]) * mode.aprime[i] + np.sin(mode.theta[i]) * mode.cprime[i]
        mode.lambda12[i] = np.cos(mode.theta[i]) * mode.bprime[i] + np.sin(mode.theta[i]) * mode.dprime[i]
        mode.lambda21[i] = -np.sin(mode.theta[i]) * mode.aprime[i] + np.cos(mode.theta[i]) * mode.cprime[i]
        mode.lambda2[i] = -np.sin(mode.theta[i]) * mode.bprime[i] + np.cos(mode.theta[i]) * mode.dprime[i]

        # Equation 32: lambdaplus, lambdaminus
        mode.lambdaplus[i] = (mode.lambda1[i] + mode.lambda2[i]) / 2.
        mode.lambdaminus[i] = (mode.lambda1[i] - mode.lambda2[i]) / 2.

        # Below eq. 37: zetaplus, zetaminus
        mode.zetaplus[i] = mode.theta[i] - mode.phi[i]
        mode.zetaminus[i] = -mode.theta[i] - mode.phi[i]

    # Below eq. 39: A0
    mode.locooffseta[0] = mode.a[0]
    mode.locooffsetc[0] = mode.c[0]

    if (DEBUG):
        print("\n\noffset:\n===============\n\n")
        print("LOCO-EFA A0 offset:\ta=%f\tc=%f\n" % (mode.locooffseta[0], mode.locooffsetc[0]))

    # Below eq. 41: A+(l=0)
    mode.locolambdaplus[0] = mode.lambdaplus[2]
    mode.locozetaplus[0] = mode.zetaplus[2]

    # Below eq. 41: A+(l=1)
    mode.locolambdaplus[1] = mode.lambdaplus[1]
    mode.locozetaplus[1] = mode.zetaplus[1]

    # Below eq. 41: A+(l>1)
    for i in range(2, N_modes):
        mode.locolambdaplus[i] = mode.lambdaplus[i + 1]
        mode.locozetaplus[i] = mode.zetaplus[i + 1]

    # Below eq. 41: A-(l>0)
    for i in range(2, N_modes + 2):
        mode.locolambdaminus[i] = mode.lambdaminus[i - 1]
        mode.locozetaminus[i] = mode.zetaminus[i - 1]
    if (DEBUG):
        print("\n\nLn quadruplets:\n===============\n\n")
        for i in range(N_modes + 2):
            print("LOCO-EFA mode %d:\tlambdaplus=%f\tlambdaminus=%f\tzetaplus=%ftzetaminus=%f\n" % (
            i, mode.locolambdaplus[i], mode.locolambdaminus[i], mode.locozetaplus[i], mode.locozetaminus[i]))

    # Equation 38: Lambda*Zeta
    mode.locoaplus = mode.locolambdaplus * np.cos(mode.locozetaplus)
    mode.locobplus = -mode.locolambdaplus * np.sin(mode.locozetaplus)
    mode.lococplus = mode.locolambdaplus * np.sin(mode.locozetaplus)
    mode.locodplus = mode.locolambdaplus * np.cos(mode.locozetaplus)
    mode.locoaminus = mode.locolambdaminus * np.cos(mode.locozetaminus)
    mode.locobminus = -mode.locolambdaminus * np.sin(mode.locozetaminus)
    mode.lococminus = -mode.locolambdaminus * np.sin(mode.locozetaminus)
    mode.locodminus = -mode.locolambdaminus * np.cos(mode.locozetaminus)
    if (DEBUG):
        print("\n\nLOCO coefficients:\n==================\n\n")
        for i in range(N_modes + 2):
            print("mode %d, Aplus:\n" % i)
            print("(%f\t%f)\n" % (mode.locoaplus[i], mode.locobplus[i]))
            print("(%f\t%f)\n" % (mode.lococplus[i], mode.locodplus[i]))
            print("mode %d, Aminus:\n" % i)
            print("(%f\t%f)\n" % (mode.locoaminus[i], mode.locobminus[i]))
            print("(%f\t%f)\n" % (mode.lococminus[i], mode.locodminus[i]))
    # Equation 47: L
    mode.locoL[1:] = np.sqrt(
        mode.locolambdaplus[1:] * mode.locolambdaplus[1:] + mode.locolambdaminus[1:] * mode.locolambdaminus[
                                                                                       1:] + 2. * mode.locolambdaplus[
                                                                                                  1:] * mode.locolambdaminus[
                                                                                                        1:] * np.cos(
            mode.locozetaplus[1:] - mode.locozetaminus[1:] - 2. * mode.locozetaplus[1]))
    if (DEBUG):
        print("\nLn scalar:\n==========\n")
        for i in range(N_modes + 2):
            print("LOCO-EFA mode %d:\tLn=%f" % (i, mode.locoL[i]))
    return mode

def reconstruct_contour(mode, tp, rec_type='EFA', first_mode=0, last_mode=2):
    N_modes = len(mode.locoL.values)
    # timepoint -= cell->contour[cellnumber][cell->contourlength[cellnumber]].t*cell->locoefa[cellnumber][1].tau/(2.*M_PI)
    if mode.r[1]<0.:
        tp = -tp
    x = [ 0. for i in tp ]
    y = [ 0. for i in tp ]
    if rec_type=='EFA':
        if first_mode == 0:
            x += mode.alpha[0]
            y += mode.gamma[0]
        for p in range(np.max([1,first_mode]), np.min([last_mode,N_modes+1])):
            x += mode.alpha[p] * np.cos(2.*np.pi*p*tp) + mode.beta[p] * np.sin(2.*np.pi*p*tp)
            y += mode.gamma[p] * np.cos(2.*np.pi*p*tp) + mode.delta[p] * np.sin(2.*np.pi*p*tp)
    elif rec_type=='LOCOEFA':
        if first_mode == 0:
            x += mode.locooffseta[0]
            y += mode.locooffsetc[0]
            # L=0
            x += mode.locolambdaplus[0] * ( np.cos(mode.locozetaplus[0]) * np.cos(2.*np.pi*2.*tp) - np.sin(mode.locozetaplus[0]) * np.sin(2.*np.pi*2.*tp) )
            y += mode.locolambdaplus[0] * ( np.sin(mode.locozetaplus[0]) * np.cos(2.*np.pi*2.*tp) + np.cos(mode.locozetaplus[0]) * np.sin(2.*np.pi*2.*tp) )
        if first_mode <= 1:
            # L=1
            x += mode.locolambdaplus[1] * ( np.cos(mode.locozetaplus[1]) * np.cos(2.*np.pi*tp) - np.sin(mode.locozetaplus[1]) * np.sin(2.*np.pi*tp) )
            y += mode.locolambdaplus[1] * ( np.sin(mode.locozetaplus[1]) * np.cos(2.*np.pi*tp) + np.cos(mode.locozetaplus[1]) * np.sin(2.*np.pi*tp) )

        # L=2...N,+
        for p in range(np.max([2,first_mode]),np.min([last_mode+1,N_modes])):
            x += mode.locolambdaplus[p] * ( np.cos(mode.locozetaplus[p]) * np.cos(2.*np.pi*(p+1)*tp) - np.sin(mode.locozetaplus[p]) * np.sin(2.*np.pi*(p+1)*tp) )
            y += mode.locolambdaplus[p] * ( np.sin(mode.locozetaplus[p]) * np.cos(2.*np.pi*(p+1)*tp) + np.cos(mode.locozetaplus[p]) * np.sin(2.*np.pi*(p+1)*tp) )

        # L=2..N,-
        for p in range(np.max([2,first_mode]),np.min([last_mode+1,N_modes+2])):
            x += mode.locolambdaminus[p] * ( np.cos(mode.locozetaminus[p]) * np.cos(2.*np.pi*(p-1)*tp) - np.sin(mode.locozetaminus[p]) * np.sin(2.*np.pi*(p-1)*tp) )
            y -= mode.locolambdaminus[p] * ( np.sin(mode.locozetaminus[p]) * np.cos(2.*np.pi*(p-1)*tp) + np.cos(mode.locozetaminus[p]) * np.sin(2.*np.pi*(p-1)*tp) )
    return x, y

def read_example_data(contour, N_modes=50):
    # read points
    if contour.x.values[-1] != contour.x.values[0]:
        print('Appending first element to make it a close curve')
        newrow = pd.Series({'x': contour.x[0], 'y': contour.y[0]})
        contour.loc[len(contour)] = newrow
    initialize_values = [0. for i in range(len(contour.x))]
    variables = ['deltax',  'deltay',       'deltat',       't',
                'xi',       'sumdeltaxj',   'sumdeltayj',   'epsilon']
    for variable in variables:
        contour[variable] = initialize_values
    mode = initialize_mode(N_modes)
    return contour, mode

def initialize_mode(N_modes_original=50):
    # construct the mode dictionary
    N_modes = N_modes_original + 2
    variables = ['alpha', 'beta', 'gamma', 'delta',
                 'tau', 'alphaprime', 'gammaprime', 'rho',
                 'alphastar', 'betastar', 'gammastar', 'deltastar',
                 'r', 'a', 'b', 'c',
                 'd', 'aprime', 'bprime', 'cprime',
                 'dprime', 'phi', 'theta', 'lambda1',
                 'lambda2', 'lambda21', 'lambda12', 'lambdaplus',
                 'lambdaminus', 'zetaplus', 'zetaminus', 'locooffseta',
                 'locooffsetc', 'locolambdaplus', 'locolambdaminus', 'locozetaplus',
                 'locozetaminus', 'locoL', 'locoaplus', 'locobplus',
                 'lococplus', 'locodplus', 'locoaminus', 'locobminus',
                 'lococminus', 'locodminus']
    initialize_values = [0. for i in range(N_modes)]
    mode = pd.DataFrame(dict(zip(variables, [initialize_values] * len(variables))))
    return mode


def metrics_calculate1(mask, image, msei_scales=[1,2,3,4,5,6], msei_wlevel=1, locoefa_nummode=50, DEBUG=True):
    '''
        计算7个类器官特征指标，透光度的计算需要image信息，其他的只需要mask
    '''
    metrics = np.zeros(7)
    contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    #area
    area = cv2.contourArea(contour)
    s = 1000 / 465
    metrics[0] = area * s
    # epi
    epi = epi_calculate(contour, area)
    metrics[1] = epi
    # msei
    msei = msei_calculate(contour, scales=msei_scales, wave_level=msei_wlevel)
    metrics[2] = msei
    # Solidity
    solid =solid_calculate(contour, area)
    metrics[3] = solid
    # Eccentricity
    eccentricity = eccentricity_calculate(contour)
    metrics[4] = eccentricity
    # Loco-efa 若允许，也可以使用mode, ln两个向量作为loco_efa结果，信息量更大
    # mode, ln, entropy = locoefa_calculate(contour, N_modes=locoefa_nummode, DEBUG=DEBUG)
    entropy = 0
    metrics[5] = entropy
    # Transmittance
    transmittance = transmittance_calculate(mask, image)
    metrics[6] = transmittance
    return metrics

def metrics_calculate(mask, image, msei_scales=[1,2,3,4,5,6], msei_wlevel=1, locoefa_nummode=50, DEBUG=True):
    '''
        计算7个类器官特征指标，透光度的计算需要image信息，其他的只需要mask
    '''
    # 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]  # 只取第一个外部轮廓

    metrics = np.zeros(7)
    # 使用并行计算加速每个特征的计算
    with ThreadPoolExecutor() as executor:
        future_area = executor.submit(cv2.contourArea, contour)  # area
        future_epi = executor.submit(epi_calculate, contour, cv2.contourArea(contour))  # epi
        future_msei = executor.submit(msei_calculate, contour, scales=msei_scales, wave_level=msei_wlevel)  # MSEI
        future_solid = executor.submit(solid_calculate, contour, cv2.contourArea(contour))  # Solidity
        future_eccentricity = executor.submit(eccentricity_calculate, contour)  # Eccentricity
        future_locoefa = executor.submit(locoefa_calculate, contour, N_modes=locoefa_nummode, DEBUG=DEBUG)  # Loco-efa
        future_transmittance = executor.submit(transmittance_calculate, mask, image)  # Transmittance
        # 获取计算结果
        metrics[0] = future_area.result()  # Area
        metrics[1] = future_epi.result()  # Epi
        metrics[2] = future_msei.result()  # MSEI
        metrics[3] = future_solid.result()  # Solidity
        metrics[4] = future_eccentricity.result()  # Eccentricity
        metrics[5] = future_locoefa.result()[2]  # Entropy, assuming locoefa returns a tuple with entropy as the third element
        metrics[6] = future_transmittance.result()  # Transmittance
    return metrics

def ensure1region(mask, min_area=50):
    '''
    保证mask中只有一个连通区域
    min_area: 设定一个最小面积阈值, 若所有连通区域都小于该值，则返回None
    '''
    # 使用 connectedComponentsWithStats 函数获取连通区域的信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    # 初始化最大面积变量
    max_area = 0
    index = 0
    # 遍历所有连通区域（从1开始，因为0是背景）
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and area > max_area:
            # 找到一个新的最大区域
            max_area = area
            index = i
    if index > 0:
        result[labels == index] = 255
        return result
    else:
        return None
def process_image(file_name, image_root, mask_root, args):
    """
    处理每个图像文件，计算相应的度量指标
    """
    # 拼接文件路径
    image_path = os.path.join(image_root, file_name)
    mask_path = os.path.join(mask_root, file_name)

    # 读取图像和掩膜
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # 设定一个最小面积阈值, 若连通区域小于该值，则跳过
    mask = ensure1region(mask, min_area=50)
    if isinstance(mask, np.ndarray):
        # 图像padding，防止图像边缘的类器官轮廓提取受影响
        pad_w = args.pad_w if hasattr(args, 'pad_w') else 2
        image = cv2.copyMakeBorder(image, pad_w, pad_w, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pad_w, pad_w, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        # 计算度量指标
        metrics = list(metrics_calculate1(mask, image, msei_scales=args.msei_scales,
                                         msei_wlevel=args.msei_wlevel,
                                         locoefa_nummode=args.locoefa_nummode,
                                         DEBUG=args.DEBUG))
        metrics.insert(0, os.path.splitext(file_name)[0])  # 添加文件名
        print(f"Finish:  {file_name} !")
        return metrics
    return None
def save_metrics(metrics_all, metrics_savepath):
    """
    将数据保存到CSV文件中
    """
    metrics_all = np.array(metrics_all)
    if metrics_all.shape[0] > 0:  # 确保 metrics_all 不为空
        # 转换为 DataFrame
        df = pd.DataFrame(metrics_all,
                          columns=['ImageName', 'Area', 'Epi', 'Msei', 'Solidity', 'Eccentricity', 'Loco-efa',
                                   'Transmittance'])
        # 使用 utf-8-sig 编码保存文件
        df.to_csv(metrics_savepath, index=False, encoding='utf-8-sig')

def main(args):
    image_root = args.image_root
    mask_root = args.mask_root
    metrics_savepath = args.metrics_savepath
    # 获取图像文件列表
    file_list = [f for f in os.listdir(image_root) if f.endswith(('jpg', 'png', 'tif'))]  # 仅处理图片文件
    # 使用多进程提高计算效率
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=cpu_count())
    # 启动进程池进行并行处理
    results = pool.starmap(process_image, [(file, image_root, mask_root, args) for file in file_list])
    # 过滤掉返回值为 None 的项
    metrics_all = [result for result in results if result is not None]
    # 关闭进程池
    pool.close()
    pool.join()
    # 将度量指标保存到 CSV 文件
    save_metrics(metrics_all, metrics_savepath)

if __name__ == '__main__':
    main(args)
