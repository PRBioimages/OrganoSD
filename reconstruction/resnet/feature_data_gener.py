import pandas as pd

# # 读取 data_ID.txt 文件中的 ID
# with open('/home/xlzhu/heying/CTCs/data_ID.txt', 'r') as file:
#     data_ids = [line.strip() for line in file]
#
# # 读取 CTC_noCTC_stats.csv 文件并筛选出相应的行
# df_base = pd.read_csv('/home/xlzhu/heying/CTCs/CTC_noCTC_stats.csv')
# df_filtered = df_base[df_base['ID'].isin(data_ids)]
#
# # 将筛选后的结果保存为 feature_data.csv 文件
# df_filtered.to_csv('/home/xlzhu/heying/CTCs/feature_data.csv', index=False)

# 读取 data_ID.txt 文件中的 ID
with open("/home/xlzhu/heying/CTCs/resnet/validdata_ID.txt", 'r') as file:
    data_ids = [line.strip() for line in file]

# 读取 CTC_noCTC_stats.csv 文件并筛选出相应的行
df_base = pd.read_csv('/home/xlzhu/heying/CTCs/CTC_noCTC_stats.csv')
df_filtered = df_base[df_base['ID'].isin(data_ids)]

# 将筛选后的结果保存为 feature_data.csv 文件
df_filtered.to_csv('/home/xlzhu/heying/CTCs/resnet/validfeature_data.csv', index=False)