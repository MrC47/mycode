# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

# 在通常训练模型时，我们会不断循环遍历数据集，直到达到了我们要求的epoch数，1个epoch是遍历一次数据集。在域泛化模型训练中，通常不是把所有训练环境数据放在一起输入模型进行训练的，而是为每个环境创建一个dataloader，在一个step中，从每个训练环境都抓取一个batch的数据，拼在一起输入到模型中去。
# 一般的采样器只支持一个数据集的循环采样。这就存在一个问题，域泛化中，我们有很多不同的域，而且是分开采样的，如果一个环境样本多，一个环境样本少，在训练过程中，我跑一个epoch，样本少的环境一个epoch很快就跑完了，但是样本多的环境却在继续。
# 举例来说，100个样本的环境，采样器会采100个索引，10个样本的环境，采样器会采10个样本，这就会出现一个现象，在1个epoch中，样本少的环境将采样器采样的数据遍历完了，但样本多的环境还在继续。
# 为了处理这个问题，我们让样本少的环境的采样器继续循环采样，使得两个环境的采样器采出来的索引一样多。
# 这样可以防止：1、如果A环境有10000张图，B环境只有100张。混在一起作为训练数据，然后洗牌，模型在一个Batch里可能全是A的图，完全忽略了B。2、一些模型训练的时候，需要这样环境分明的情况。且这样处理，可以支持一些复杂的DG算法。
# 在DomainBed中，我们通常同时从多个环境（Domains）中读取数据。环境A可能有1000张图。环境B可能只有100张图。如果我们按标准的一个Epoch训练，环境B很快就跑完了，而环境A还没过半。为了让训练能持续进行（通常按Steps而不是Epochs计），我们需要让环境B跑完后立即“重头开始”。
# 在通常训练模型时，由于只有一个数据集，我们会不断循环遍历数据集，直到达到了我们要求的epoch数，模型的输入可以理解为一个由n个epoch数据组成的大数据集。
# 在这里，我们遍历n个epoch，模型的输入同样可以理解为一个由n个epoch数据组成的大数据集，但是这个数据集可以分为k个域，每个域中，一个epoch所包含的样本量取决于样本量最大的那个环境。
# 例如，训练环境A有100个样本，训练环境B有10个样本，那么一个epoch中，有200个样本，其中环境A样本遍历1遍，环境B样本遍历10遍。
# 在DomainBed的实际代码实现中，它其实不再计算“一个Epoch总共有多少样本”，它只关心**“我要跑多少个 Iterations（Steps）”**。也就是说，在DomainBed中，没有像一般模型那样一个epoch一个epoch那么明确的训练了。可以说是epoch的边界模糊了。
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
# 传入的是BatchSampler对象，BatchSampler对象可以理解为二维列表，这个列表中的每个元素是一个列表，每个二级列表中的元素是batchsize个索引值。例如,batchsize为4，则[[5,3,6,2],[11,34,45,64],[14,53,63,35]]。
# 普通的Sampler对象则就是一个一维列表，每个元素就是索引值。普通Sample对象没有batchsize参数。
            for batch in self.sampler:
# 学习yield是怎么回事。
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()
# weights传入的是一个与数据集长度相等的一维列表（或 Tensor）。列表中的每一个数字，对应数据集中每一个样本的“中奖概率权重”。数字越大代表这个样本在抽样时被选中的概率就越高。数字越小代表这个样本被选中的概率就越低。
# 有权重随机采样会使某类样本或者某几个样本在sample采样的索引中重复出现。
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)
# DataLoader对象本身不是迭代器（Iterator），它是一个可迭代对象（Iterable）。
# 可迭代对象就是像列表、字典那样，你可以遍历它。将一个可迭代对象传入一个iter（）函数，该函数就返回一个迭代器，你只有对返回的迭代器调用next（）函数，它才会返回一个可迭代对象中的元素，当到了末尾，就会抛出StopIteration异常。
# 可迭代对象可以重复遍历，但是迭代器只能遍历一次，遍历完就报废了。其实，迭代器就是在可迭代对象中加入一个指针，刚开始指向可迭代对象的第一个元素，调用next（）后，返回所指的元素并将指针移向下一个元素。故迭代器只能遍历一次。
# 可迭代对象必须有__iter__，迭代器必须有__next__ 和 __iter__，一个类只有有__next__方法，才能作为构建迭代器的原料，否则只有__iter__只能作为一个可迭代对象。
# 在一般的训练中，往往是for batch in dataloader，背后其实发生了两件事：1、Python自动帮你调用了it = iter(dataloader)创建了一个临时迭代器。2、每次循环自动帮你调用 next(it)。3、当遍历完一个epoch，循环结束，这个临时迭代器就被销毁了。4、第二个epoch开始，则生成一个新的迭代器。
# 当你写self._infinite_iterator = iter(torch.utils.data.DataLoader时，是你自己构造了一个迭代器，这个迭代器不会自动销毁。其作用是防止样本量少的环境遍历完一个epoch之后自动把迭代器销毁产生问题。
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class InfiniteDataLoaderWithoutReplacement:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError



class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
