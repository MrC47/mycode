# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os

import tqdm

from domainbed.lib.query import Q

# 这段代码是DomainBed实验分析流程的“入口点”。它的任务是从磁盘中将成千上万条实验记录拉取到内存中，并将其转化为Q对象。
# DomainBed 的实验结果通常是这样存储的：
# output_dir/
# ├── experiment_0/
# │   └── results.jsonl
# ├── experiment_1/
# │   └── results.jsonl
# └── ...
# 代码遍历path下的每一个子目录subdir。每个子目录通常代表一个特定的参数组合（比如某种算法在某个种子下的运行结果）。
# （看json和jsonl的区别）results.jsonl是JSON Lines格式。与普通的.json 不同，它每一行都是一个独立的JSON对象。这样做的好处是：1、鲁棒性：如果实验中途崩溃，之前已经写入磁盘的行依然是有效的。2、内存友好：读取时可以一行一行处理，而不需要一次性加载一个巨大的数组。

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
# line[:-1]是为了去掉行尾的换行符\n。
                    records.append(json.loads(line[:-1]))
# 在跑大规模实验（Sweep）时，有些任务可能因为集群故障、显存溢出或被手动中断而没有生成results.jsonl。如果这里不加try...except，程序会直接死掉。有了它，程序会跳过那些失败的实验文件夹，只收集成功完成的数据。这保证了分析脚本的健壮性。
        except IOError:
            pass
# return Q(records)：这是最关键的一步。它把原本只是普通列表的records升级成了具备强力搜索和统计能力的Q对象。
    return Q(records)

# 一条实验记录（Record）可能包含多个测试环境的结果，但我们需要按“单一测试环境”来汇总分析。例如，在原始数据里一条记录里包含test_envs: [0, 1]。我们需要分别统计“当0是测试集时，算法表现如何”以及“当1是测试集时，算法表现如何”。
#  假如一条实验结果为：
# {
#     "args": {
#         "algorithm": "ERM",
#         "trial_seed": 100,
#         "test_envs": [0, 1]
#     },
#     "step": 1000,
#     "env0_out_acc": 0.85,
#     "env1_out_acc": 0.72,
#     "env2_out_acc": 0.65,
#     "env3_out_acc": 0.90
# }
# 经过处理后，由于有两个测试环境的实验，这一条实验结果的每个环境的实验结果，都会按照这个函数的要求，被重构成一个新的条目，由于有两个测试环境，那么会分裂为两条记录。
def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]:
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env)
            result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()])
