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

class ErmPlusPlusMovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 600
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone())/(1.+self.sma_count))
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



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
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
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

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

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

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

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

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

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
