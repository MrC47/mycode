# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Save some representative images from each dataset to disk.
"""
import random
import torch
import argparse
from domainbed import hparams_registry
from domainbed import datasets
import imageio
import os
from tqdm import tqdm

# 这段代码是一个典型的数据可视化/调试脚本。它的目的不是训练模型，而是从各个数据集的每个环境中随机抽取样本并保存为图片，以便开发者直观地检查数据是否加载正确、预处理（如归一化）是否正确生效。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
# 解析命令行中输入的参数。执行完这一行后，你原本输入的那一串杂乱的文本，就变成了代码里可以点出来的属性：args.data_dir。
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets_to_save = ['OfficeHome', 'TerraIncognita', 'DomainNet', 'RotatedMNIST', 'ColoredMNIST', 'SVIRO']

    for dataset_name in tqdm(datasets_to_save):
# 获取数据集在ERM算法下的默认超参数。
        hparams = hparams_registry.default_hparams('ERM', dataset_name)
# 获取dataset_name对应的数据集类，并实例化。
        dataset = datasets.get_dataset_class(dataset_name)(
            args.data_dir,
            list(range(datasets.num_environments(dataset_name))),
            hparams)
        for env_idx, env in enumerate(tqdm(dataset)):
# 每个环境随机抽50张图片查看。
            for i in tqdm(range(50)):
                idx = random.choice(list(range(len(env))))
                x, y = env[idx]
# 过滤逻辑：如果标签大于10（比如第11类以后的），就重新抽
# 这是为了防止在类别太多的数据集（如DomainNet）中抽到太偏的类
                while y > 10:
                    idx = random.choice(list(range(len(env))))
                    x, y = env[idx]
# 补全通道：如果是2通道数据（极少见），补一个0矩阵凑成3通道 (RGB)。
# torch.zeros_like(x)和x拼起来是4通道，因为x是两通道，故我们只要前三通道，即[:3,:,:]。
# 在PyTorch张量中，x的维度顺序通常是[通道, 高, 宽]。
                if x.shape[0] == 2:
                    x = torch.cat([x, torch.zeros_like(x)], dim=0)[:3,:,:]
# 逆归一化：如果像素值有负数（min < 0），说明之前减去了均值。
                if x.min() < 0:
# 使用ImageNet的标准均值和标准差进行还原。
# [:,None,None]是一个非常精妙的Python/PyTorch维度扩展（Unsqueeze）技巧。它的意思是：在这个位置增加一个长度为1的新维度。
# 原始的mean是一维，而图像是三维，故需要维度扩展。之所以要把mean从[3]变成[3, 1, 1]，是为了触发PyTorch的广播机制。
# 当执行x * std + mean时，x的形状是[3, 224, 224]，mean的形状是 [3, 1, 1]，PyTorch会自动把mean的那两个长度为1的维度“拉伸”成 224。
# mean的第0个通道的值（0.485），会加到x第0个通道的所有224x224个像素上。以此类推，处理第1和第2个通道。如果不加None, None，程序就不知道这3个数字是应该横着加、竖着加、还是按通道加。
                    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
                    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
# 公式：原始值 = (处理值 * 标准差) + 均值。
                    x = (x * std) + mean
                    assert(x.min() >= 0)
                    assert(x.max() <= 1)
# 像素还原：将[0, 1] 映射回[0, 255]。
# x之所以落在0到1之间，是因为在datasets.py加载数据时，PyTorch的transforms.ToTensor() 已经对其进行了预处理。这个操作做了两件大事：1、改变维度：将图片从[H, W, C]（高度, 宽度, 通道）转为[C, H, W]。2、归一化(Scaling)：将所有像素值除以255，把原本0到255的整数变成了0.0到1.0的浮点数。
# 为什么不直接乘以255，而是用255.99呢？这主要涉及到浮点数转整数时的“截断”机制以及数值稳定性。
# 在计算机中，当你把一个浮点数（float）强制转换成整数（int/uint8）时，它是向下取整（Floor），而不是四舍五入。
# 如果你用255，假设某个像素还原后的最大值理论上应该是1.0。但在计算机浮点数运算中，由于精度限制，它算出来的结果可能是0.99999994。0.99999994 * 255 = 254.99998，强转uint8后，它变成了254。你本该得到最亮的白色（255），结果却因为微小的精度误差损失了一个色阶。
# 为什么不是256？因为那样会溢出。如果一个像素真的是完美的 1.0，1.0 * 256 = 256.0，对于uint8（范围 0-255）来说，256溢出了，它会绕回0（变成纯黑色）。这样你的图片上就会出现莫名其妙的黑色斑点。
                x = (x * 255.99)
# .astype('uint8'): 转为 8 位无符号整型（图片标准格式）。
# .transpose(1,2,0): 把 [C, H, W] 转为 [H, W, C] (因为绘图库只认后者)。
                x = x.numpy().astype('uint8').transpose(1,2,0)
# 构造文件名：包含数据集名、环境名、样本序号、原始索引、类别ID，并使用imageio库将处理好的数组保存为磁盘上的图片文件。
                imageio.imwrite(
                    os.path.join(args.output_dir,
                        f'{dataset_name}_env{env_idx}{dataset.ENVIRONMENTS[env_idx]}_{i}_idx{idx}_class{y}.png'),
                    x)
