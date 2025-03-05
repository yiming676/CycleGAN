import torch
import itertools
import tensorboardX
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from models import Discriminator, Generator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset

# 为了确保主程序的入口点
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchsize = 1
    size = 256
    lr = 0.0002
    n_epoch = 200
    epoch = 0
    decay_epoch = 100

    # 定义网络
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    # 定义损失函数
    loss_GAN = torch.nn.MSELoss()
    loss_cycle = torch.nn.L1Loss()
    loss_identity = torch.nn.L1Loss()

    # 定义优化器
    opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.9999))
    opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.9999))
    opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.9999))

    # 定义学习率调度器
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)

    # 数据路径、日志路径
    data_root = r"E:\E_AlgorithmStudy\Python\datasets\apple2orange"
    log_path = "logs"
    input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    label_real = torch.ones([1], requires_grad=False).to(device)
    label_fake = torch.zeros([1], requires_grad=False).to(device)

    # 定义缓冲区
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 定义数据增强
    transforms_ = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # 定义数据加载器
    dataloader = DataLoader(
        ImageDataset(data_root, transforms_),
        batch_size=batchsize,
        shuffle=True,
        num_workers=2  # 数据加载使用主线程
    )

    writer_log = tensorboardX.SummaryWriter(log_path)
    step = 0

    # 训练循环
    for epoch in range(n_epoch):
        for i, batch in enumerate(dataloader):
            real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
            real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

            # 训练生成器
            opt_G.zero_grad()

            same_B = netG_A2B(real_B)
            loss_identity_B = loss_identity(same_B, real_B) * 5.0

            same_A = netG_B2A(real_A)
            loss_identity_A = loss_identity(same_A, real_A) * 5.0

            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = loss_GAN(pred_fake, label_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = loss_GAN(pred_fake, label_real)

            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0

            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            loss_G.backward()
            opt_G.step()

            # 训练判别器 A
            opt_DA.zero_grad()
            pred_real = netD_A(real_A)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = loss_GAN(pred_real, label_fake)  # 注意这里使用了 pred_real，应该是使用假样本的 pred_fake 才对

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            opt_DA.step()

            # 训练判别器 B
            opt_DB.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_B = fake_A_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_real, label_fake)  # 同上，这里是使用了 pred_real，应该是使用假样本的 pred_fake 才对

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_DB.step()

            # 每隔 100 个步骤记录一次日志
            if step % 100 == 0:
                print("loss_G:{:.4f}, loss_G_identity:{:.4f}, loss_G_GAN:{:.4f}, "
                      "loss_G_cycle:{:.4f}, loss_D_A:{:.4f}, loss_D_B:{:.4f}".format(
                    loss_G.item(), (loss_identity_A + loss_identity_B).item(),
                    (loss_GAN_A2B + loss_GAN_B2A).item(),
                    (loss_cycle_BAB + loss_cycle_ABA).item(),
                    loss_D_A.item(), loss_D_B.item()
                ))

                writer_log.add_scalar("loss_G", loss_G.item(), global_step=step + 1)
                writer_log.add_scalar("loss_G_identity", (loss_identity_A + loss_identity_B).item(), global_step=step + 1)
                writer_log.add_scalar("loss_G_GAN", (loss_GAN_A2B + loss_GAN_B2A).item(), global_step=step + 1)
                writer_log.add_scalar("loss_G_cycle", (loss_cycle_BAB + loss_cycle_ABA).item(), global_step=step + 1)
                writer_log.add_scalar("loss_D_A", loss_D_A.item(), global_step=step + 1)
                writer_log.add_scalar("loss_D_B", loss_D_B.item(), global_step=step + 1)

            step += 1

        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(netG_A2B.state_dict(), "models/netG_A2B.pth")
            torch.save(netG_B2A.state_dict(), "models/netG_B2A.pth")
            torch.save(netD_A.state_dict(), "models/netD_A.pth")
            torch.save(netD_B.state_dict(), "models/netD_B.pth")

    # 关闭日志写入器
    writer_log.close()