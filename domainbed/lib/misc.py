# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import copy
import math
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
from collections import Counter
from itertools import cycle

#计算两个神经网络参数空间的
def distance(h1, h2):
    ''' distance of two networks (h1, h2 are classifiers)'''
    dist = 0.
# state_dict()是PyTorch里的一个字典，存着模型每一层的名字（Key）和对应的权重张量（Value）。这里遍历的是参数的名字。
    for param in h1.state_dict():
# 根据刚才拿到的名字 param，分别去h1和h2中取出对应的数值张量。
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
# 计算两个张量之间每一个对应元素的差值。并计算范数。对于矩阵，默认计算的是Frobenius范数。
# 为什么取平方？因为欧几里得距离的公式是 $\sum (x_1 - x_2)^2$。我们先把每一层差异的平方和累加到 dist 里。
        dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
# 对累加完的平方和进行开根号。即欧式距离。
    return torch.sqrt(dist)
# 实现欧几里得投影，通俗的说，该函数的作用是如果一个模型跑得太远了，就把它拉回到以原模型为中心、半径为δ的球体内。” 这在对抗训练（Adversarial Training）和约束优化中非常常见。
# 对抗训练中，对抗者努力修改模型的参数，想方设法让模型变得最糟糕（比如让 Loss 最大化），训练者的任务是调整模型，让模型变得最稳定。
# 如果不加限制，这个“鼠”会把模型参数改得面目全非（比如把权重改成无穷大），那模型直接就坏掉了，这种训练也就失去了意义。
# 为什么要限制在“球体”内？1、保护模型的基本功能：我们希望看到模型在“受到一点点扰动”时是否依然稳健。如果扰动太大，那就不是“测试稳健性”，而是“拆迁”了。2、模拟真实世界的波动：在域泛化（Domain Bed）中，我们假设不同领域（比如照片和素描）之间的差异是有限的。通过在δ范围内制造对抗样本，是模拟在合理范围内的环境变化。
# 在约束优化中的意义：很多时候，我们寻找的最优解必须满足某些条件（比如参数不能太大，否则会过拟合）。当你更新一步参数后，发现新参数出界了。这时候投影的目的就是：在边界上找一个离出界点最近的点。这就是为什么代码里要算 ratio（比例）——它是为了保证拉回来的时候，方向不变，只缩短距离。
# delta是允许最大半径，adv_h是尝试跑远的模型，h是中心点模型。
def proj(delta, adv_h, h):
    ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
    ''' adv_h and h are two classifiers'''
    dist = distance(adv_h, h)
    if dist <= delta:
        return adv_h
    else:
# 如果超出了范围，就要进行缩放。ratio是一个小于1的比例系数。比如允许距离是10，现在跑到了20，那么ratio就是 0.5。
        ratio = delta / dist
# 使用zip同时遍历两个模型的每一层参数。
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
# 投影公式，(param_adv_h - param_h)是算出从中心点指向远方的向量。ration*（）用于把这个距离缩短。param_h + ...表示从中心点出发，沿着缩短后的向量走一段。这样param_adv_h被强行拉回到了距离param_h正好等于delta的那个球面上。
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # print("distance: ", distance(adv_h, h))
        return adv_h
# 计算两个字典（通常存储的是模型的参数 state_dict）之间的L2距离（均方误差）。
# 两个字典dict_1和dict_2。在PyTorch中，这通常是model.state_dict()，键是层名，值是权重张量。
def l2_between_dicts(dict_1, dict_2):
# 检查两个字典的长度是否相等。如果两个模型结构不同（比如一个有10层，一个有5层），计算就没有意义了。如果长度不等，程序会直接报错并停止。
    assert len(dict_1) == len(dict_2)
# 按Key的字母顺序提取所有参数。sotred（）非常重要，因为字典在Python中虽然是有序的，但为了绝对保险，必须手动排序。这样能确保dict_1的第一层权重对应的是dict_2的第一层权重，而不是错位去减第二层。
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
# 把每一层的矩阵拉直成一条一维向量。把这些拉直后的线头尾相接，拼成一根极长的大向量。然后对每一维的差值求平方，最后再平方的平均值，即最后算出的是均方误差MSE。
# torch.cat接收的是一个元组或列表。作者在这里显式地转换成tuple是为了符合PyTorch早期的语法规范，也是一种非常稳健的写法。
# 这个函数在DomainBed中通常用来监控训练的稳定性：如果（两个epoch或者当前模型与初始状态的）l2_between_dicts的值突然变得巨大，说明模型参数正在发生剧烈震荡。又三种解决方案：1、调小学习率。2、增加权重衰减。3、使用模型平均。
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

# 这段代码实现的是深度学习中的一种技术：移动平均（Moving Average），具体在这里叫SMA (Simple Moving Average)。
# 其核心思想是与其相信训练过程中某一个瞬间的模型，不如把过去一段时间的模型全部加起来取平均。 这样得到的模型通常更稳健，泛化能力更强。
class ErmPlusPlusMovingAvg:
    def __init__(self, network):
        self.network = network
# 复制一份网络，这个网络就是用来存储“平均后的参数”的影子模型。
        self.network_sma = copy.deepcopy(network)
# 将影子模型设为评估模式。因为它不负责训练，只负责被平均。
        self.network_sma.eval()
# 设置一个“等待期”。前 600 步模型还不稳定，先不进行平均。
        self.sma_start_iter = 600
# 记录总步数。
        self.global_iter = 0
# 记录参与平均的次数。
        self.sma_count = 0
# 这个函数通常在每个训练步（Step）之后被调用一次。
    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
# 过了等待期（开始算平均），使用zip同时拉起两个模型的参数：param_q是当前正在跳动的原始模型，param_k是之前的平均模型。name是一个字符串，代表神经网络中每一个具体参数（权重或偏置）的“身份证号”或“路径”
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
# 排除掉BatchNorm层里的一些统计参数，这些不需要手动平均。
# 在神经网络中，有一种非常常用的层叫Batch Normalization (批归一化，简称 BN 层)。BN层除了有需要学习的权重（weight）和偏置（bias）外，还需要记录一些“统计数据”来帮助它工作。num_batches_tracked就是其中之一。
# 它本质上是一个“计数器”，记录了这个BN层一共处理了多少个batch的数据。num_batches_tracked是一个整数（LongTensor）。
# 在PyTorch的底层定义中，模型参数分为两类：1、Parameters (参数)：比如权重和偏置。它们是浮点数，通过梯度下降进行更新。2、Buffers (缓存)：比如BN层的均值（running_mean）、方差（running_var）和这个num_batches_tracked。它们不通过梯度更新，但需要被保存。
# 如果不写这行判断，代码会尝试对它做移动平均计算，这会导致两个严重问题：1、类型冲突 (Type Mismatch)，num_batches_tracked是整数。当你把它除以(1. + count)（浮点数）时，结果会变成浮点数。如果你尝试把一个浮点数填回一个本该是整数的坑位，PyTorch可能会报错。
# 2、逻辑错误 (Logic Error)：num_batches_tracked的含义是“处理过的batch总数”。影子模型（SMA）处理了N个 batch。原始模型处理了N个 batch。对它们求平均值没有任何意义。正确的做法是让它们各自记录自己的计数，或者直接保留原始模型的计数。
# 为什么不排除BN里的均值和方差呢？因为均值和方差代表了模型在训练过程中观察到的数据的“统计特征”。它们参与了前向传播的数学计算。既然是浮点数，它们在数学上是可以进行加权平均计算的，不会产生类型错误。
                if 'num_batches_tracked' not in name:
# 使用的是增量平均，假设之前平均了n次，现在加入第n+1个模型，new_avg = old_sum + new_val / (n+1)
# old_sum = param_k*n,param_k是已经算好的均值，为了参与n+1次的平均，故要乘n。
# 为什么不直接求和，而是用均值乘n呢？1、防止数值溢出。2、满足即时可用性，这段代码的设计思路是让network_sma随时随地都是一个可以直接拿来测试的有效模型。如果存的是总和，模型就“没法即时可用了”，必须手动除以n才能恢复成正常的权重。
                    new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone())/(1.+self.sma_count))
        else:
# 处在等待期内，直接把原始模型network的参数复制给network_sma。这相当于让影子模型先跟着跑，直到跑稳了再开始平均。
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)

# 这段代码实现的是指数移动平均（Exponential Moving Average, EMA）。之前的是算数平均。EMA给近期的数据分配更高的权重，给很久以前的数据分配较低的权重。
class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
# 衰减系数（通常设为 0.99 或 0.999）。它代表了“记忆”的强度。
        self.ema = ema
# 一个字典，用来存储每一层参数平滑后的结果。
        self.ema_data = {}
        self._updates = 0
# 这是一个缩放修正开关，目的是为了保持数值的量级。
        self._oneminusema_correction = oneminusema_correction
# dict_data通常是当前Step的模型参数字典（state_dict）。
    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
# 拉平成一行，即一维向量。
            data = data.view(1, -1)
            if self._updates == 0:
# 如果以前没平均果，则赋零值
                previous_data = torch.zeros_like(data)
            else:
# 如果有进行平均，就读取之前的值。
                previous_data = self.ema_data[name]
# ema越大，模型对过去记忆力越强，更新越迟钝（更平滑）。ema越小，则模型对当前更敏感。
            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
# 这里就是指数移动平均里的偏差修正。只不过这里是简单粗暴地每一轮都除以（1-ema）。
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


# 为类别不平衡的数据集生成采样权重。如果某个类别的数据特别少，就给它更高的权重，让模型在训练时有更多机会“看到”它。这种方法被称为加权采样（Weighted Sampling）。
def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
# 计算每个类别的权重，样本越少，权重越大。
        weight_per_class[y] = 1 / (counts[y] * n_classes)
# 根据刚才算好的各类权重，给数据集里的每一个样本贴上属于它的权重标签。返回一个和数据集等长的PyTorch张量weights。
    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

# 一个用于调试的函数，设置断点，因为再命令行中，没有图形界面，讲这个函数加在你觉得有问题的代码行之前。
def pdb():
# 在很多科研框架中，为了记录日志，程序会把sys.stdout（标准输出）重定向到文件（比如 output.txt）或者某个网页UI上。如果你在调试，交互界面的提示符（(Pdb)）也会被发往文件，你根本看不见，也就没法输入命令。
# 这一行强行把输出拨回到真正的控制台终端（sys.__stdout__ 代表系统原始的、未被修改过的输出流）。
    sys.stdout = sys.__stdout__
    import pdb
# 为什么提示输入'n'？：因为当你调用这个自定义的pdb()函数时，断点是停在这个函数内部的。输入n（next）或up才能跳回到你真正想调试的那行代码。
# Pdb 常用指令快捷表：1、l，查看当前断点周围的代码内容。2、p，打印变量的值。3、n，执行下一行代码。4、s，进入函数内部（如果你想看某个子函数的逻辑）。5、c，退出调试模式，让程序继续正常运行。6、q，强制结束整个程序。
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

# 这段代码是一个非常稳健的随机数种子生成器。它的作用是：无论你给它什么输入（数字、字符串、列表等），它都能将其转化为一个固定范围内的整数，作为随机种子的“根”。
# *args这是一个Python的语法糖，代表接收任意数量、任意类型的参数。你可能会把文件名、迭代次数、超参数同时传进去。只要这些输入不变，生成的种子就永远不变。
def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
# .encode("utf-8")：将字符串转为字节流（二进制），因为哈希算法不直接处理字符串。
# .hexdigest()：将 MD5 算出来的二进制结果转为一个16进制的字符串。
# ，16：配合 int() 函数，告诉Python：“请把这个16进制的字符串转成一个巨大的十进制整数。"
# % (2**31)将那个巨大的整数限制在0到2^{31}-1之间。因为在很多系统（尤其是 C 语言底层）中，随机数种子的最大值通常是32位整数的上限。超过这个范围可能会导致程序报错或溢出。
# 为什么要这么麻烦？直接用random.seed(42)不行吗？因为在DomainBed这种大规模实验框架里，简单的固定种子是不够用的。假设你在跑10个不同的实验，每个实验的learning_rate不同。如果你在代码里硬编码seed = 42，那么这10个实验的随机初始化竟然是一模一样的。
# 使用这种方式，为每个实验生成不同的种子，既保证了实验之间的独立性，又保证了如果你以后用同样的参数重跑，结果一定能复现。
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

# 这段代码定义了一个非常轻量级的类，它的核心作用是：在不复制原始数据的前提下，为一个庞大的数据集创建一个“虚拟视图”或“切片”。在DomainBed这种需要把数据集拆分为训练集（Train）和验证集（Validation）的场景中，这个类非常高效。
# 它继承了PyTorch标准的Dataset类。这意味着它的实例可以像普通数据集一样被放入DataLoader中进行迭代。
class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
# underlying_dataset：指的是“底层数据集”（即那个还没拆分的完整大数组）。
        self.underlying_dataset = underlying_dataset
# keys：这是一个索引列表（例如 [0, 2, 5, 10...]）。它记录了属于这个子集的样本在原图中的位置。
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

# 将数据集按参数n切分为两部分，用于训练集和验证集。
# 为什么 DomainBed 要这样拆分？在域泛化（Domain Generalization）实验中，我们通常需要从每一个领域（Domain）中拿出一部分数据做验证。而且，为了公平对比，必须保证验证集的划分是随机的，但又是可复现的。
# 测试集呢？在DomainBed中，并不是没有测试集，而是通过一种**“留一法”（Leave-one-out）**的逻辑来定义测试集。即留一个域作为测试集，剩下的域，每个域的数据，一部分作为训练集，一部分作为验证集。
def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

# 它是许多**域对齐（Domain Alignment）**算法（如VREx, Fish, SD等）的核心。它的作用是：从多个不同的领域中随机两两配对，构造出“对比组”。
# minibatches：这是一个列表，里面每个元素代表一个领域（Domain）的一个Batch。例如，如果有3个领域，列表长度就是 3。
def random_pairs_of_minibatches(minibatches):
# randperm：生成一个随机排列。比如有 3 个领域，可能会生成 [2, 0, 1]。打乱领域的顺序，确保每一轮训练时，领域之间的配对都是随机的。
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
# 实现所有领域能够两两配对，不重不漏。例如有[A，B，C]三个领域，则i=A，j=B、i=B，j=C，i=C，j=A。
        j = i + 1 if i < (len(minibatches) - 1) else 0
# 取出第i个领域的数据和标签，取出第j个领域的数据和标签。
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
# 对齐长度：不同领域的Batch Size可能因为数据量不平衡而略有不同。为了做减法或对比运算，必须确保两组数据的样本数量一致。
        min_n = min(len(xi), len(xj))
# 将截取到相同长度的两组数据打包成一个pair。
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

# 这段代码实现的是元学习（Meta-Learning）算法（如 MLDG, MetaReg 等）在域泛化中的核心逻辑。
# 它与刚才看到的“随机配对”非常相似，但有一个本质的区别：它将领域划分为“元训练集”（Meta-Train）和“元测试集”（Meta-Test）。
# 也就是输入是一系列的pair，即pairs数组，先使用这些数组中的元组的第一位，即训练数据进行训练，然后再使用第二维，即测试数据进行测试。
# 为什么不手动划分几个领域作为训练集，一个领域作为测试集呢，非要搞那么麻烦。其实，“手动划分”和代码里的“麻烦操作”同时存在，但它们发生在不同的层级，服务于不同的目的。
# 简单来说：手动划分是为了“大考”，而MLDG的pairs是为了“模拟考”。即在训练集中，再分测试集。这是MLDG的内容了。
def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]
# cycle(meta_test)：这是一个无限循环迭代器。如果meta_test只有1个域（比如域 C），而meta_train有 3 个域（A, B, D），它会产生：(A, C), (B, C), (D, C)。
# 让每一个参与“训练”的领域都去和那个“模拟测试”的领域进行配对。
    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

# 一个计算准确率的函数。
def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

# 这段代码实现了一个非常经典且实用的功能：“镜像输出”（Output Mirroring）。
# 在Linux命令行中有一个命令叫tee，它的作用是把数据像字母“T”的形状一样分开：一端流向屏幕（屏幕显示），另一端流向文件（保存记录）。这个类就是在用Python模拟这个功能。
# 在train.py的开头，你通常会看到这样一行：ys.stdout = Tee(os.path.join(args.output_dir, 'out.txt'))，即同时把输出发向屏幕和文件。
# 与pdb()函数的联系：如果你用了Tee，那么sys.stdout已经变成了一个“双向水管”。你想进入PDB调试时，PDB这种复杂的交互式工具无法在Tee这个自定义类里正常工作。所以pdb()第一步必须先把sys.stdout还原回系统原始的sys.__stdout__，这样你才能在屏幕上看到(Pdb)提示符并输入命令。
class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)
# 为什么要flush？通常系统为了提高性能，会把要打印的内容先存进内存缓冲区。如果你不Flush，万一程序突然崩溃（比如OOM显存溢出），缓冲区里的报错信息可能还没来得及写进磁盘就丢失了。强制Flush保证了日志的实时性。
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
# 这段代码定义了一个非常强大的工具类ParamDict。它的核心逻辑是：把一个存放模型参数的“字典”，伪装成一个可以进行数学运算的“向量”或“数字”。
# 在元学习（Meta-Learning，如 MLDG、Reptile）中，我们需要频繁地对整套模型参数进行加减乘除，传统的做法是写循环遍历字典，而这个类让你能像写普通数学题一样操作整套权重。
# 个类继承自OrderedDict（有序字典），但它重写了Python的算术运算符（+, -, *, /）。
class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
# 运算原型，这是内部的“中央处理器”。当你执行运算时，它会判断：如果是数字，就把字典里的每一个Tensor都和这个数字做运算。如果是另一个字典，就把两个字典中“相同名字”的 Tensor 拿出来做对位运算。
    def _prototype(self, other, op):
# 判断另一个参与运算的“数”是不是纯数字，如果是，则
# {k: op(v, other) for k, v in self.items()}是一个字典推导式。
# self.items()：遍历当前模型的所有层。k是层名（如 'layer1.weight'），v 是这一层的Tensor张量。
# op(v, other)：对这一层的张量v执行指定的运算（如 add 或 mul），操作数就是那个数字other。
# PyTorch的广播机制：这里利用了 PyTorch 的特性。当你用Tensor + 5时，PyTorch会自动把这个5加到张量里的每一个元素上。
# ParamDict(...)：将生成的新字典重新包装成ParamDict类，这样你得到的结果依然拥有“数学超能力”，可以继续进行下一步运算。
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError
# operator.add是Python内置operator模块中的一个函数版本的加法运算符。operator.add(a, b)和你平时写的a + b在功能上是完全等价的。
    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


############################################################
# A general PyTorch implementation of KDE. Builds on:
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py
############################################################

class Kernel(torch.nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bw=None):
        super().__init__()
        self.bw = 0.05 if bw is None else bw

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        if dims == ():
            x_sq = diffs ** 2
        else:
            x_sq = torch.norm(diffs, p=2, dim=dims) ** 2

        var = self.bw ** 2
        exp = torch.exp(-x_sq / (2 * var))
        coef = 1. / torch.sqrt(2 * np.pi * var)

        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        # device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bw
        return train_Xs + noise

    def cdf(self, test_Xs, train_Xs):
        mus = train_Xs                                                      # kernel centred on each observation
        sigmas = torch.ones(len(mus), device=test_Xs.device) * self.bw      # bandwidth = stddev
        x_ = test_Xs.repeat(len(mus), 1).T                                  # repeat to allow broadcasting below
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x_))


def estimate_bandwidth(x, method="silverman"):
    x_, _ = torch.sort(x)
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)

    if method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method.lower() == 'gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)

    else:
        raise ValueError(f"Invalid method selected: {method}.")

    return bandwidth


class KernelDensityEstimator(torch.nn.Module):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel='gaussian', bw_select='Gauss-optimal'):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.train_Xs = train_Xs
        self._n_kernels = len(self.train_Xs)

        if bw_select is not None:
            self.bw = estimate_bandwidth(self.train_Xs, bw_select)
        else:
            self.bw = None

        if kernel.lower() == 'gaussian':
            self.kernel = GaussianKernel(self.bw)
        else:
            raise NotImplementedError(f"'{kernel}' kernel not implemented.")

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_Xs)


############################################################
# PyTorch implementation of 1D distributions.
############################################################

EPS = 1e-16


class Distribution1D:
    def __init__(self, dist_function=None):
        """
        :param dist_function: function to instantiate the distribution (self.dist).
        :param parameters: list of parameters in the correct order for dist_function.
        """
        self.dist = None
        self.dist_function = dist_function

    @property
    def parameters(self):
        raise NotImplementedError

    def create_dist(self):
        if self.dist_function is not None:
            return self.dist_function(*self.parameters)
        else:
            raise NotImplementedError("No distribution function was specified during intialization.")

    def estimate_parameters(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        return self.create_dist().log_prob(x)

    def cdf(self, x):
        return self.create_dist().cdf(x)

    def icdf(self, q):
        return self.create_dist().icdf(q)

    def sample(self, n=1):
        if self.dist is None:
            self.dist = self.create_dist()
        n_ = torch.Size([]) if n == 1 else (n,)
        return self.dist.sample(n_)

    def sample_n(self, n=10):
        return self.sample(n)


def continuous_bisect_fun_left(f, v, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


class Normal(Distribution1D):
    def __init__(self, location=0, scale=1):
        self.location = location
        self.scale = scale
        super().__init__(torch.distributions.Normal)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        var = sum([(x_i - mean) ** 2 for x_i in x]) / (len(x) - 1)
        self.location = mean
        self.scale = torch.sqrt(var + EPS)

    def icdf(self, q):
        if q >= 0:
            return super().icdf(q)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            return self.location + self.scale * math.sqrt(-2 * log_y)


class Nonparametric(Distribution1D):
    def __init__(self, use_kde=True, bw_select='Gauss-optimal'):
        self.use_kde = use_kde
        self.bw_select = bw_select
        self.bw, self.data, self.kde = None, None, None
        super().__init__()

    @property
    def parameters(self):
        return []

    def estimate_parameters(self, x):
        self.data, _ = torch.sort(x)

        if self.use_kde:
            self.kde = KernelDensityEstimator(self.data, bw_select=self.bw_select)
            self.bw = torch.ones(1, device=self.data.device) * self.kde.bw

    def icdf(self, q):
        if not self.use_kde:
            # Empirical or step CDF. Differentiable as torch.quantile uses (linear) interpolation.
            return torch.quantile(self.data, float(q))

        if q >= 0:
            # Find quantile via binary search on the KDE CDF
            lo = torch.distributions.Normal(self.data[0], self.bw[0]).icdf(q)
            hi = torch.distributions.Normal(self.data[-1], self.bw[-1]).icdf(q)
            return continuous_bisect_fun_left(self.kde.cdf, q, lo, hi)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            v = torch.mean(self.data + self.bw * math.sqrt(-2 * log_y))
            return v

# --------------------------------------------------------
# LARS optimizer, implementation from MoCo v3:
# https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


############################################################
# Supervised Contrastive Loss implementation from:
# https://arxiv.org/abs/2004.11362
############################################################
class SupConLossLambda(torch.nn.Module):
    def __init__(self, lamda: float=0.5, temperature: float=0.07):
        super(SupConLossLambda, self).__init__()
        self.temperature = temperature
        self.lamda = lamda

    def forward(self, features: torch.Tensor, labels: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _ = features.shape
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        # create a lookup table for pairwise dot prods
        pairwise_dot_prods = torch.matmul(normalized_features, normalized_features.T)/self.temperature
        loss = 0
        nans = 0
        for i, (label, domain_label) in enumerate(zip(labels, domain_labels)):

            # take the positive and negative samples wrt in/out domain            
            cond_pos_in_domain = torch.logical_and(labels==label, domain_labels == domain_label) # take all positives
            cond_pos_in_domain[i] = False # exclude itself
            cond_pos_out_domain = torch.logical_and(labels==label, domain_labels != domain_label)
            cond_neg_in_domain = torch.logical_and(labels!=label, domain_labels == domain_label)
            cond_neg_out_domain = torch.logical_and(labels!=label, domain_labels != domain_label)

            pos_feats_in_domain = pairwise_dot_prods[cond_pos_in_domain]
            pos_feats_out_domain = pairwise_dot_prods[cond_pos_out_domain]
            neg_feats_in_domain = pairwise_dot_prods[cond_neg_in_domain]
            neg_feats_out_domain = pairwise_dot_prods[cond_neg_out_domain]


            # calculate nominator and denominator wrt lambda scaling
            scaled_exp_term = torch.cat((self.lamda * torch.exp(pos_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(pos_feats_out_domain[:, i])))
            scaled_denom_const = torch.sum(torch.cat((self.lamda * torch.exp(neg_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(neg_feats_out_domain[:, i]), scaled_exp_term))) + 1e-5

            # nof positive samples
            num_positives = pos_feats_in_domain.shape[0] + pos_feats_out_domain.shape[0] # total positive samples
            log_fraction = torch.log((scaled_exp_term / scaled_denom_const) + 1e-5) # take log fraction
            loss_i = torch.sum(log_fraction) / num_positives
            if torch.isnan(loss_i):
                nans += 1
                continue
            loss -= loss_i # sum and average over num positives
        return loss/(batch_size-nans+1) # avg over batch
