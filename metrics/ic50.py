import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser( description='Calculate IC50')
parser.add_argument('--root', default='/home/hryang/Detecseg/metric/ic50/',
                    type=str, help='存储单类器官特征的的文件夹目录，里面有对照组(命名为conctrol)以及不同浓度药物下类'
                                   '器官特征的csv文件，还有药物浓度的csv文件(命名为concentrations)')
parser.add_argument('--savepath', default='/home/hryang/Detecseg/sig_metric/nasal.csv',
                    type=str, help='计算结果的存储路径')
parser.add_argument('--feature_weights', default=[1, 0, 0, 0, 0, 0, 0],
                    type=list, help='各个特征的权重系数')
parser.add_argument('--initial_guess', default=[100, 1, 100000, 0],
                    type=list, help='初始化猜测参数A,B,C,D')
args = parser.parse_args()


def calculate_mean_without_outliers(data, threshold=3):
    means = []
    for col in range(data.shape[1]):
        column_data = data[:, col]
        mean = np.mean(column_data)
        std_dev = np.std(column_data)
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev
        filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
        filtered_mean = np.mean(filtered_data)
        means.append(filtered_mean)
    return np.array(means)


def four_param_logistic(x, A, B, C, D):
    '''
    四参数Logistic回归模型
    Y: 细胞存活率（%）
    X: 药物浓度（μM）
    A: 最大反应值（通常为100%）
    B: 曲线的斜率
    C: IC50值（药物浓度对应50%存活率）
    D: 最小反应值（通常为0%）
    '''
    return (A - D) / (1 + (x / C)**B) + D

def sigmoid(x, A, B, C, D):
    return D + (A - D) / (1 + (x / C) ** B)


def main(args):
    # 文件读取
    control = pd.read_csv(os.path.join(args.root, 'control.csv'))
    concentrations = pd.read_csv(os.path.join(args.root, 'concentrations.csv'))
    labels = concentrations.columns.tolist()

    feature_data = []
    for label in labels:
        data = pd.read_csv(os.path.join(args.root, label))
        feature_data.append(data.to_numpy())

    # 对照组数据处理
    control = np.dot(calculate_mean_without_outliers(control.to_numpy()), args.feature_weights)

    # 细胞生存率数据计算
    cell_viability_data = []
    for data in feature_data:
        mean_data = calculate_mean_without_outliers(data)
        cell_viability_data.append(np.dot(mean_data, args.feature_weights) / control * 100)

    concentrations = concentrations.to_numpy().flatten()
    cell_viability_data = np.array(cell_viability_data)

    params, covariance = curve_fit(four_param_logistic, concentrations, cell_viability_data, p0=args.initial_guess)
    A_fit, B_fit, C_fit, D_fit = params
    print(f"拟合的参数: A = {A_fit}, B = {B_fit}, C (IC50) = {C_fit}, D = {D_fit}")
    IC50 = C_fit
    print(f"IC50值: {IC50} μM")

    # 可视化拟合结果
    plt.scatter(concentrations, cell_viability_data, label="Experimental Data", color='red')
    x_fit = np.linspace(min(concentrations), max(concentrations), 100)
    y_fit = four_param_logistic(x_fit, *params)
    plt.plot(x_fit, y_fit, label="Fitted Curve", color='blue')
    plt.xlabel('Drug Concentration (μM)')
    plt.ylabel('Cell Viability (%)')
    plt.legend()
    plt.title('Dose-Response Curve')
    plt.show()

if __name__ == '__main__':
    main(args)