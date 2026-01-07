# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex

# 这段代码定义了一个Job类，它是DomainBed自动化实验管理（Sweep）的核心。如果说train.py负责跑一个实验，那么Job类就负责管理、生成和追踪成百上千个实验。
# 它解决了一个痛点：如何自动化地生成实验命令、给每个实验分配唯一的文件夹，并记住哪些实验跑完了，哪些没跑完。
# 例如，你正在写论文，需要测试3种算法（ERM, DANN, CORAL），在4个数据集上，每个实验重复3次随机种子。那么你需要进行36个实验。手动一个一个进行实验是非常麻烦的，还容易忘。
# 有了Job类，你只需要给它一个列表：algorithms = [ERM, DANN]。Job类会自动帮你拼装：python -m domainbed.scripts.train --algorithm ERM --dataset OfficeHome ...跑完所有实验。
# 此外，它还会给每个实验生成唯一文件夹，检查实验状态。
class Job:
# 这是一种状态机设计，它通过检查文件的状态来判断任务进度，这样即使程序崩溃重启，它也能断点续传。
    NOT_LAUNCHED = 'Not launched' # 文件夹还没创建。
    INCOMPLETE = 'Incomplete' # 文件夹有了，但没跑完，程序可能跑一半崩溃了。
    DONE = 'Done' # 程序跑完了。

    def __init__(self, train_args, sweep_output_dir):
# 这一段代码通过哈希给每个实验生成唯一id，并给每个实验创建唯一的文件夹。
# 这段代码把所有的训练参数train_args转成JSON字符串。使用hashlib.md5计算哈希值。
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
# 开始构建命令行。
        command = ['python', '-m', 'domainbed.scripts.train']
# 遍历所有参数，把它们转成命令行参数格式（--key value）。
        for k, v in sorted(self.train_args.items()):
# 如果值是列表（如测试环境 [1, 2]），转成空格分隔的字符串 "1 2"。
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
# 这是一个非常专业的细节。如果你有一个文件夹叫My Data（带空格），普通的写法拼出来会变成 --data_dir My Data（被识别成两个参数），而shlex会把它变成 --data_dir 'My Data'。
# 如果字符串很“干净”（只有数字、字母、下划线），shlex.quote就什么都不做。
# 如果发现空格、括号、引号、分号等，如果发现空格、括号、引号、分号等，shlex.quote会自动给整个字符串套上单引号 '...'。
# 如果字符串内部本来就有单引号，它还会聪明地进行转义。
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
# 把列表拼成最终要在终端执行的长字符串。
        self.command_str = ' '.join(command)

# 状态检查，如果文件夹里有done文件，说明跑完了。有文件夹但没done文件，说明中断了。连文件夹都没有，就是还没开始。
        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

# 当你print(job)时，显示的友好提示信息。
    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
# 复制一份任务列表。
        jobs = jobs.copy()
# 随机打乱任务顺序（防止大量重任务堆积在一起）。
        np.random.shuffle(jobs)
        print('Making job directories:')
# 为所有任务建好文件夹。
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
# 调用发射函数（如 command_launcher.py里的函数）真正去执行这些命令。
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

# 它的作用是为拥有n个环境的数据集，自动计算出所有可能的“测试集组合”。如果n=3，那么最终输出为：[0]（环境0为测试环境，1，2则为训练环境）, [0, 1], [0, 2], [1], [1, 2], [2]。
# 为什么要这么设计？DomainBed的论文中提到，为了严谨，不能只测一个环境。有时测试模型在一个未知域上的表现（单域测试）。有时测试模型在多个未知域上的鲁棒性（双域测试）。
def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

# 生成代办任务清单，如果你要跑1000个实验，那么这个函数就要写出这1000个实验的具体参数。
# n_trials:试验次数（不同的 trial_seed）。通常为了结果可靠，同一个实验会换不同的随机种子跑多次。
# n_hparams:超参数搜索的数量。DomainBed 不只跑默认参数，还会自动随机生成几十组超参数来寻找最优解。
# single_test_envs:布尔值。如果是True，每次只测一个环境；如果是False，则调用你之前看到的all_test_env_combinations测遍所有组合。
def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
    data_dir, task, holdout_fraction, single_test_envs, hparams):
    args_list = []
# 这个段代码有多重循环。
# 第一层：trial_seed（保证实验可重复性）。
# 第二层：dataset（在哪些数据集上跑，如OfficeHome）。
# 第三层：algorithm（用什么算法，如ERM, Mixup）。
# 第四层：test_envs（选哪些环境当测试集）。
# 第五层：hparams_seed（选哪一组随机超参数）。
# 总任务数 = n_trials * 数据集数 * 算法数 * 测试环境组合数 * 超参数组数
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if single_test_envs:
                    all_test_envs = [
                        [i] for i in range(datasets.num_environments(dataset))]
                else:
                    all_test_envs = all_test_env_combinations(
                        datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

# 这段代码定义了一个安全拦截函数，用于在执行具有“破坏性”或“不可逆”操作（例如：删除所有实验记录、格式化数据、启动大规模扣费集群任务）之前，要求用户进行人工确认。
def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)
# 它的目的是：从所有可用的数据集中，自动剔除掉那些专门用于测试代码逻辑的“调试版（Debug）”数据集，只留下真正用于科研和论文实验的数据集。
DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams
    )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
# 自动过滤：只发射那些“还没开始”的任务。已经跑完或跑了一半的不会被重复发射。
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
# 安全拦截：除非你加了 --skip_confirmation，否则它会停下来问你 Are you sure?
        if not args.skip_confirmation:
            ask_for_confirmation()
# 查找发射器：根据你输入的发射器名称，去注册表里找对应的执行函数。
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
# 专门针对那些“卡住”或者“崩溃”的任务。
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
# 删除那些跑一般崩溃的任务的文件夹。腾出硬盘空间，下次launch时它们就会变成NOT_LAUNCHED重新跑。
        Job.delete(to_delete)
