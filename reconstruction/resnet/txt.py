# import os
#
# input_file = '/home/xlzhu/heying/CTCs/data.txt'
# output_file = '/home/xlzhu/heying/CTCs/resnet/data_noID.txt'
#
# with open(input_file, 'r') as file:
#     lines = file.readlines()
#
# new_lines = []
# for line in lines:
#     image_path, _, label = line.strip().split(' ')
#     new_line = f"{image_path} {label}\n"
#     new_lines.append(new_line)
#
# with open(output_file, 'w') as file:
#     file.writelines(new_lines)
#
# import os
#
# input_file = '/home/xlzhu/heying/CTCs/train.txt'
# output_file = '/home/xlzhu/heying/CTCs/resnet/traindata_noID.txt'
#
# with open(input_file, 'r') as file:
#     lines = file.readlines()
#
# new_lines = []
# for line in lines:
#     image_path, _, label = line.strip().split(' ')
#     new_line = f"{image_path} {label}\n"
#     new_lines.append(new_line)
#
# with open(output_file, 'w') as file:
#     file.writelines(new_lines)


import os

input_file = '/home/xlzhu/heying/CTCs/valid.txt'
output_file = '/home/xlzhu/heying/CTCs/resnet/validdata_noID.txt'

with open(input_file, 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    image_path, _, label = line.strip().split(' ')
    new_line = f"{image_path} {label}\n"
    new_lines.append(new_line)

with open(output_file, 'w') as file:
    file.writelines(new_lines)
