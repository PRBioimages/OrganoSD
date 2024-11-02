train_file = "/home/xlzhu/heying/CTCs/train.txt"
valid_file = "/home/xlzhu/heying/CTCs/valid.txt"
output_dir = "/home/xlzhu/heying/CTCs/resnet/"

# 读取 train.txt 文件，保留每行的第二项
with open(train_file, 'r') as file:
    train_lines = [line.strip().split(' ')[1] + '\n' for line in file]

# 将处理后的结果保存到 traindata_ID.txt 文件中
train_output_file = output_dir + "traindata_ID.txt"
with open(train_output_file, 'w') as file:
    file.writelines(train_lines)

# 读取 valid.txt 文件，保留每行的第二项
with open(valid_file, 'r') as file:
    valid_lines = [line.strip().split(' ')[1] + '\n' for line in file]

# 将处理后的结果保存到 validdata_ID.txt 文件中
valid_output_file = output_dir + "validdata_ID.txt"
with open(valid_output_file, 'w') as file:
    file.writelines(valid_lines)
