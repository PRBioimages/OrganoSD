
from reconstruction.ResAE.utils import *
import tqdm
import pandas as pd
from sklearn.metrics import recall_score
from reconstruction.ResAE.configs import Config
import torch
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from reconstruction.ResAE.losses.regular import FocalLoss
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report

#绘制训练损失和验证损失的曲线图
# def plt_losses(train_loss, val_loss, train_loss_rec, train_loss_class, val_loss_rec, val_loss_class, savepath):
# # def plt_losses(train_loss, val_loss, savepath):
#     plt.figure(figsize=(10, 8))
#     plt.semilogy(train_loss, label='Train')
#     plt.semilogy(val_loss, label='Valid')
#     plt.semilogy(train_loss_rec, label='Train Loss Rec')
#     plt.semilogy(train_loss_class, label='Train Loss Class')
#     plt.semilogy(val_loss_rec, label='Valid Loss Rec')
#     plt.semilogy(val_loss_class, label='Valid Loss Class')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Loss')
#     # plt.grid()
#     plt.legend()
#     # plt.title('loss')
#     # plt.show()
#     plt.savefig(savepath)
#     plt.clf()
#     plt.close()


#要的训练函数，负责执行模型的训练和验证
def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print('[ √ ] Basic training')
    train_losses = []
    val_losses = []
    train_losses_rec = []
    train_losses_class = []
    val_losses_rec = []
    val_losses_class = []
    try:
        optimizer.zero_grad()
        #每个训练轮次（epoch）中，它执行以下操作
        for epoch in range(cfg.train.num_epochs):
            # first we update batch sampler if exist
            #学习率调度器
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            #模型设置为训练模式
            model.train()
            weight_class = 0
            if epoch >= 20:
                weight_class = 5
            elif epoch >= 40:
                weight_class = 10

            #使用 tqdm 迭代器 tq 遍历训练数据加载器 train_dl
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []


            #对每个批次的图像数据 image_batch 和标签 lbl 执行以下操作
            for i, image_batch in enumerate(tq):
                image_batch = image_batch.cuda()
                # image_batch = image_batch
                # lbl = lbl
                #学习率热身（warm-up）
                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)
                # image_batch, lbl = image_batch.cuda(), lbl.cuda()

                #前向传播，将图像数据输入到模型中，得到模型的输出
                decoded_data = model(image_batch)
                
                # loss = loss_func(output, lbl)
                #计算损失，使用损失函数 loss_func 对模型的输出和输入image_batch 进行比较，得到损失值 loss
                loss_rec = loss_func(decoded_data, image_batch)
                # print(model.class_out.shape, lbl.shape)
                #loss_function = FocalLoss(gamma=2)
                #loss_class = loss_function(model.class_out, lbl)
                #loss_function = torch.nn.CrossEntropyLoss()
                #loss_class = loss_function(model.class_out, torch.argmax(lbl, dim=1))
                # print(loss_class)
                # print(loss_rec)
                loss = loss_rec

                train_losses_rec.append(loss_rec.item())
                #train_losses_class.append(loss_class.item())
                #losses.append(loss.item())


                #反向传播损失，计算梯度，并执行参数更新
                loss.backward()
                # predicted.append(output.detach().sigmoid().cpu().numpy())
                # truth.append(lbl.detach().cpu().numpy())

                #如果满足优化器的步骤条件（cfg.optimizer.step），则执行参数更新，并清零梯度
                if i % cfg.optimizer.step == 0:
                    if cfg.train.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                #如果使用学习率调度器（scheduler）为 CyclicLR、OneCycleLR 或 CosineAnnealingLR，则更新学习率。
                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                    # TODO maybe, a bug
                        scheduler.step()
                #如果 tune 为 None，在每个批次结束时更新 tqdm 进度条以显示当前损失和学习率
                if not tune:
                    info = {'loss': np.array(losses).mean(), 'lr':optimizer.param_groups[0]['lr']}
                    tq.set_postfix(info)

            #在每个训练轮次结束后，它执行以下操作
            #保存当前模型的权重到指定的路径，以便后续恢复训练或推理。
            torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
            ## 验证模型，并计算验证损失。此外，生成图像对比以供分析
            fig_path = os.path.join(save_path, 'figure')
            validate_loss, val_losses_class, val_losses_rec = basic_validate(model, valid_dl, loss_func, cfg, os.path.join(fig_path, 'plotoutput'), writer, epoch)
            
            val_losses.append(validate_loss)
            train_losses.append(np.array(losses).mean())

            #记录训练损失和验证损失，并绘制损失曲线图，保存为图片
            #plt_losses(train_losses, val_losses, train_losses_rec, train_losses_class, val_losses_rec, val_losses_class, os.path.join(fig_path, 'loss.png'))

            #使用 Tensorboard 的写入器 writer 记录训练和验证损失以进行可视化
            print(('[ √ ] epochs: {}, train loss: {:.4f}, valid loss: {:.4f}').format(
                epoch, np.array(losses).mean(), validate_loss))
            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('valid_f{}/val_loss'.format(cfg.experiment.run_fold), validate_loss, epoch)
            writer.add_scalar('train_f{}/loss_rec'.format(cfg.experiment.run_fold), np.mean(train_losses_rec), epoch)
            writer.add_scalar('train_f{}/loss_class'.format(cfg.experiment.run_fold), np.mean(train_losses_class), epoch)
            writer.add_scalar('valid_f{}/loss_rec'.format(cfg.experiment.run_fold), np.mean(val_losses_rec), epoch)
            #writer.add_scalar('valid_f{}/loss_class'.format(cfg.experiment.run_fold), np.mean(val_losses_class), epoch)


            #将训练和验证结果的信息写入日志文件，包括当前轮次、学习率、训练损失和验证损失
            with open(save_path / 'train.log', 'a') as fp:
                fp.write('{}\t{:.8f}\t{:.4f}\t{:.4f}\n'.format(
                    epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean(), validate_loss))

            #如果使用学习率调度器 scheduler 且采用 ReduceLROnPlateau 调度器，根据验证损失来动态调整学习率
            if scheduler and cfg.scheduler.name in ['ReduceLROnPlateau']:
                scheduler.step(validate_loss)
    #如果用户通过 Ctrl + C 中断了训练过程，它会捕获 KeyboardInterrupt 异常，并保存当前模型的权重，以便后续继续训练
    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        torch.save(model.state_dict(), save_path / 'checkpoints/quit_f{}.pth'.format(cfg.experiment.run_fold))

#保存原始图像和经过解码后的图像，以便在训练期间或训练完成后进行可视化和分析
def save_img(img, decoder_img, savepath, n=5):
    print(img.shape)
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=4, ncols=n, sharex=True, sharey=True, figsize=(n*4, 12))
    axes = axes.ravel()
    for idx in np.arange(2 * n):
        ax = fig.add_subplot(4, n, idx + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(decoder_img[0][idx], (1, 2, 0)))
        axes[idx].axis('off')

    #首先创建一个图形，其中包含两行，每行包含 n 个图像。第一行显示解码后的图像，而第二行显示原始图像。
    # 每个图像都以特定的大小和布局显示，然后通过 plt.savefig 函数保存为文件
    # plot the first ten input images and then reconstructed images
    # fig, axes = plt.subplots(nrows=2, ncols=n, sharex=True, sharey=True, figsize=(36, 12))
    for idx in np.arange(2 * n):
        ax = fig.add_subplot(4, n, idx + 1 + n*2, xticks=[], yticks=[])
        plt.imshow(np.transpose(img[0][idx], (1, 2, 0)))
        axes[2 * n + idx].axis('off')

    plt.savefig(savepath)
    plt.clf()
    plt.close()

#在验证集上评估模型的性能
#mdl: 要评估的模型，dl: 验证数据加载器

def basic_validate(mdl, dl, loss_func, cfg, save_path, writer, epoch):
    # weight_class = 0
    # if epoch >= 20:
    #     weight_class = 5
    # elif epoch >= 50:
    #     weight_class = 10
    #将模型切换到评估（evaluation）模式，以便在评估时不计算梯度
    mdl.eval()
    val_losses_rec = []
    val_losses_class = []
    with torch.no_grad():
        #创建空列表 results 以保存评估结果
        results = []
        #losses-每个批次的损失值，predicted-模型的预测结果
        # predicted_p-模型的预测概率值，truth-真实标签或真实数值（ground truth）
        losses, predicted, predicted_p, truth = [], [], [], []
        #遍历验证数据加载器 dl 中的每个批次（批次中包含图像 ipt 和标签 lbl）
        for i, (ipt, lbl) in enumerate(dl):
            # ipt, lbl = ipt.cuda(), lbl.cuda()
            ipt, lbl = ipt.cuda(), lbl.cuda()

            #对每个批次的图像数据进行前向传播，得到模型的输出图像 cell
            cell = mdl(ipt)
            # loss = loss_func(cell, ipt)
            loss_rec = loss_func(cell, ipt)
            loss_function = torch.nn.CrossEntropyLoss()
            loss_class = loss_function(mdl.class_out, torch.argmax(lbl, dim=1))
           # print(loss_class)
            loss = loss_class

            
            if not len(loss.shape) == 0:
                loss = loss.mean()

            #记录该批次的损失到列表 losses中
            val_losses_rec.append(loss_rec.item())
            #val_losses_class.append(loss_class.item())
            losses.append(loss.item())
            pred_labels = torch.argmax(mdl.class_out, dim=1).cpu().numpy().tolist()
            predicted_p.append(F.softmax(mdl.class_out, dim=1).cpu().numpy().max().tolist())
            predicted.append(pred_labels)
            #predicted.append(output.cpu().numpy())
            truth.append(torch.argmax(lbl, dim=1).cpu().numpy().tolist())
            

            if i < 10:
                print('mdl.class_out', F.softmax(mdl.class_out, dim=1))
                print('lbl', lbl.cpu().numpy())
                print('pred', pred_labels)
            

            # 计算准确度
            #accuracy = ((predicted > 0.5) == truth).sum().astype(np.float) / truth.shape[0] / truth.shape[1]  # 计算准确度
            #accuracies.append(accuracy)
            
            results.append({
                'step': i,
                'loss': loss.item(),
            })
            #保存第一个批次的输入图像（ipt）和模型输出图像（cell）以进行可视化
            if i == 0:
                # cell = cell.view(cfg.train.batch_size, 3, 96, 96)
                cell_numpy = cell.cpu().detach().numpy()
                ipt_numpy = ipt.cpu().numpy()
             #  save_img(ipt_numpy, cell_numpy, os.path.join(save_path, f'{epoch}.png'))


        #计算所有批次的损失的均值，得到验证集的平均损失 val_loss
        val_loss = np.array(losses).mean()
        val_loss_class = np.array(val_losses_class).mean()
        val_loss_rec = np.array(val_losses_rec).mean()

        #accuracy = ((predicted > 0.5) == truth).sum().astype(np.float) / truth.shape[0] / truth.shape[1]

        #auc = macro_multilabel_auc(truth, predicted, gpu=0)
        print('truth', truth)
        # 将模型输出转换为预测类别
        # pred_labels = torch.argmax(mdl.class_out, dim=1).cpu().numpy()
        print('pred_labels', predicted)
        print('predicted_p', predicted_p)

        # 计算召回率、准确率和F1指标
        truth, predicted= np.array(truth), np.array(predicted)
        recall = recall_score(truth, predicted, average='macro')
        accuracy = accuracy_score(truth, predicted)
        f1 = f1_score(truth, predicted, average='macro')
        report = classification_report(truth, predicted)

        # 打印评估报告
        print(report)

        # 将指标记录到 TensorBoard
        writer.add_scalar('valid_f{}/recall'.format(cfg.experiment.run_fold), recall, epoch)
        writer.add_scalar('valid_f{}/accuracy'.format(cfg.experiment.run_fold), accuracy, epoch)
        writer.add_scalar('valid_f{}/f1'.format(cfg.experiment.run_fold), f1, epoch)
        

        return val_loss, val_loss_class, val_loss_rec

