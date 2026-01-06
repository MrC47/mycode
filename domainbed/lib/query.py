# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""Small query library."""

import collections
import inspect
import json
import types
import unittest
import warnings
import math

import numpy as np

# 在DomainBed跑完实验后，它会生成一个巨大的字典（就像一个复杂的嵌套盒子），记录了所有的运行结果。
# 如果你想拿到“训练准确率”和“测试准确率”，你得手动写很多代码，例如，train_acc = results["train_stats"]["acc"]，test_acc = results["test_stats"]["acc"]
# 如果你有100个指标要选，或者你的路径是动态变化的，代码就会变得非常臃肿。
# 这个函数允许你只用一个“简单的路径字符串”来自动生成取数逻辑。这个函数是一个工厂函数，将输入的字符串转变成一个lambda表达式，我们可以通过这个返回的lambda表达式，来提取想要的内容。
# 如果输入是单一字符，则返回执行obj[key]操作的lambda函数。如果输入是“k1.k2”，则返回执行obj[k1][k2]操作的lambda函数。如果输入是“k1,k2”，则返回执行(obj[k1], obj[k2])操作的lambda。
# 如果输入的是“k1.k2,k3.k4”，文章通过递归，实现了执行obj[k1][k2],obj[k3][k4]的函数。
# 使用这个函数，将这个字符串变成函数。这样，无论你想看哪个指标，都不需要修改Python源码，只需要改传入的参数即可。
def make_selector_fn(selector):
    """
    If selector is a function, return selector.
    Otherwise, return a function corresponding to the selector string. Examples
    of valid selector strings and the corresponding functions:
        x       lambda obj: obj['x']
        x.y     lambda obj: obj['x']['y']
        x,y     lambda obj: (obj['x'], obj['y'])
    """
    if isinstance(selector, str):
        if ',' in selector:
            parts = selector.split(',')
            part_selectors = [make_selector_fn(part) for part in parts]
            return lambda obj: tuple(sel(obj) for sel in part_selectors)
        elif '.' in selector:
            parts = selector.split('.')
            part_selectors = [make_selector_fn(part) for part in parts]
            def f(obj):
                for sel in part_selectors:
                    obj = sel(obj)
                return obj
            return f
        else:
            key = selector.strip()
            return lambda obj: obj[key]
    elif isinstance(selector, types.FunctionType):
        return selector
    else:
        raise TypeError

# 在Python中，只有“不可变”的对象（如字符串、整数、元组）才是可哈希的，而“可变”的对象（如字典、列表）是不可哈希的。
# 这段代码的核心目的是，无论你给它什么对象，它都要把它转换成一个可以被“哈希”（Hashable）的形式。
# 在DomainBed这种框架里，经常需要把实验参数（通常是一个复杂的字典）作为“键（Key）”存进缓存或者用来生成唯一的标识符。如果你直接把{'lr': 0.01, 'model': 'resnet'}当做字典的 Key，Python会报错：TypeError: unhashable type: 'dict'。
# 这段代码首先尝试进行哈希，若不成功，就把obj转换成json字符串，并设置强制让字典里的键按字母顺序排列。变成字符串后就可哈希了。
# 此外，强制排序也可以防止数据重复。假设有两个字典，内容一模一样，但顺序不同：字典A: {"lr": 0.01, "batch": 32}， 字典B: {"batch": 32, "lr": 0.01}
# 如果不排序，它们生成的JSON字符串可能不一样，导致系统认为这是两个不同的实验配置。有了sort_keys=True，无论原始数据顺序如何，生成的字符串永远是 "batch": 32, "lr": 0.01}。
def hashable(obj):
    try:
        hash(obj)
        return obj
    except TypeError:
        return json.dumps({'_':obj}, sort_keys=True)

# 该类本质上是一个增强版的列表，专门用来对海量的实验结果（JSON 字典列表）进行类似 SQL 数据库一样的查询、分组和统计操作。如果你跑了10次实验，那么就会有一个含有10个元素的列表，每个元素是一个实验结果的字典。
class Q(object):
    def __init__(self, list_):
        super(Q, self).__init__()
        self._list = list_

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        return self._list[key]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._list == other._list
        else:
            return self._list == other

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return repr(self._list)

    def _append(self, item):
        """Unsafe, be careful you know what you're doing."""
        self._list.append(item)

# logs = [
#         {"metadata": {"alg": "ERM", "ds": "PACS"}, "acc": 70},
#         {"metadata": {"alg": "ERM", "ds": "PACS"}, "acc": 72},
#         {"metadata": {"alg": "MLDG", "ds": "PACS"}, "acc": 80},
#         {"metadata": {"alg": "ERM", "ds": "OfficeHome"}, "acc": 60}
#     ]
# 如果传入selector='metadata.alg,metadata.ds'，经过处理后，会返回一个Q对象，包含3个元组（按Key排序后）：
    # (('ERM', 'OfficeHome'), Q([第4条记录]))
    # (('ERM', 'PACS'), Q([第1条, 第2条记录]))
    # (('MLDG', 'PACS'), Q([第3条记录]))
# 如果传入的是selector='metadata.alg,metadata.ds'，则返回的Q对象包含：
# group函数的本质是数据降维与结构化。
    def group(self, selector):
        """
        Group elements by selector and return a list of (group, group_records)
        tuples.
        """
        selector = make_selector_fn(selector)
        groups = {}
        for x in self._list:
            group = selector(x)
            group_key = hashable(group)
            if group_key not in groups:
                groups[group_key] = (group, Q([]))
            groups[group_key][1]._append(x)
        results = [groups[key] for key in sorted(groups.keys())]
        return Q(results)

    def group_map(self, selector, fn):
        """
        Group elements by selector, apply fn to each group, and return a list
        of the results.
        """
        return self.group(selector).map(fn)

    def map(self, fn):
        """
        map self onto fn. If fn takes multiple args, tuple-unpacking
        is applied.
        """
        if len(inspect.signature(fn).parameters) > 1:
            return Q([fn(*x) for x in self._list])
        else:
            return Q([fn(x) for x in self._list])

    def select(self, selector):
        selector = make_selector_fn(selector)
        return Q([selector(x) for x in self._list])

    def min(self):
        return min(self._list)

    def max(self):
        return max(self._list)

    def sum(self):
        return sum(self._list)

    def len(self):
        return len(self._list)

    def mean(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.mean(self._list))

    def std(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.std(self._list))

    def mean_std(self):
        return (self.mean(), self.std())

    def argmax(self, selector):
        selector = make_selector_fn(selector)
        return max(self._list, key=selector)

    def filter(self, fn):
        return Q([x for x in self._list if fn(x)])

    def filter_equals(self, selector, value):
        """like [x for x in y if x.selector == value]"""
        selector = make_selector_fn(selector)
        return self.filter(lambda r: selector(r) == value)

    def filter_not_none(self):
        return self.filter(lambda r: r is not None)

    def filter_not_nan(self):
        return self.filter(lambda r: not np.isnan(r))

    def flatten(self):
        return Q([y for x in self._list for y in x])

    def unique(self):
        result = []
        result_set = set()
        for x in self._list:
            hashable_x = hashable(x)
            if hashable_x not in result_set:
                result_set.add(hashable_x)
                result.append(x)
        return Q(result)

    def sorted(self, key=None):
        if key is None:
            key = lambda x: x
        def key2(x):
            x = key(x)
            if isinstance(x, (np.floating, float)) and np.isnan(x):
                return float('-inf')
            else:
                return x
        return Q(sorted(self._list, key=key2))
