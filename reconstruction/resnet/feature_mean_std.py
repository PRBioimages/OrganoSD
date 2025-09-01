import pandas as pd

# 从CSV文件中加载数据
csv_file_path = '/home/xlzhu/heying/CTCs/resnet/trainfeature_data.csv'
data = pd.read_csv(csv_file_path)

# 提取所需特征列
selected_columns = ['5-dapi_area', '8-dapi_mean', '9-ck_area', '12-ck_mean', '13-ck_total',
                     '15-cd45_mean', '22-dapi_fb_mean', '23-ck_fb_mean', '24-cd45_fb_mean',
                     '26-ck_vs_cd45', '28-ck_dapi_Area', '30-ck_impurity_fb']

selected_data = data[selected_columns]

# 计算均值和标准差
mean_values = selected_data.mean()
std_values = selected_data.std()

# 创建一个DataFrame保存结果
result_df = pd.DataFrame({
    'Feature': selected_columns,
    'Mean': mean_values,
    'Std': std_values
})

# 保存到CSV文件
result_csv_path = '/home/xlzhu/heying/CTCs/resnet/trainfeature_mean_std.csv'
result_df.to_csv(result_csv_path, index=False)

# 打印结果DataFrame
print(result_df)
