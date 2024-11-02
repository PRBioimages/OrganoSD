import torch.optim.lr_scheduler as lr_scheduler

def adjust_learning_rate(optimizer, epoch, initial_lr, milestones, gamma):
    """根据epoch调整学习率"""
    lr = initial_lr
    for milestone in milestones:
        if epoch >= milestone:
            lr *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 示例用法
if __name__ == '__main__':
    # 定义优化器和初始学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_lr = 0.1

    # 定义里程碑和衰减因子
    milestones = [40, 80]
    gamma = 0.1

    # 在训练循环中调用学习率调整函数
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, initial_lr, milestones, gamma)
        # 进行训练
