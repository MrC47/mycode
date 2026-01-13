# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

# vars（object）：返回对象object的属性和属性值的字典对象。object可以是类，也可以是python模块（文件）。
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

# 这一段代码利用了python的反射功能，反射，就是是指程序在运行时（Runtime）能够“观察”并“修改”自身结构和行为的能力。简单点说，就是代码可以根据一个“字符串名字”去找到对应的变量、函数或类。
# 普通调用：dataset = datasets.OfficeHome(...)
# 反射调用：getattr(datasets, "OfficeHome")(...)
    if args.dataset in vars(datasets):
# 从字典中取出所选定的类，并实例化。
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.

    # 它的目的是将每一个环境（Domain）的数据切分成不同的子集，以支持“领域泛化（DG）”和“无监督领域自适应（UDA）”两种任务的公平对比。
    # in_splits存每个环境的训练部分（主要的学习来源）。
    # out_splits存每个环境的验证部分（用于模型选择，类似开发集）。
    # uda_splits存测试环境中的无标签数据（仅用于领域自适应任务）。
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
# 将当前环境 (env) 切分为 out 和 in。
# args.holdout_fraction通常是0.2，表示 20%拿出来做验证(out)，80%留着做训练(in_)。
# misc.seed_hash 确保每个环境的切分顺序是由trial_seed决定的，保证可重复性。
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
# 该部分针对UDA。
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
# 如果超参数要求类别平衡（防止某些类别样本太多导致偏差）。
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))
# 如果你明确要求做“领域自适应”任务，但结果没切出 UDA 数据，报错。
    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

# 可以看到，train_loadrs是一个列表，里面的每一个元素都是InfiniteDataLoader，一个环境一个InfiniteDataLoader。
# 在构建的train_loaders的时候，用到了列表推导，并把测试集排除在外了。if i not in args.test_envs
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

# 为什么验证集还要用到in_splits、out_splits和uda_splits，不是光用out_splits就行了吗？
# 深度学习里，我们不仅要关心模型学得好不好，还要关心它学没学进去。
# 为什么要测模型已经看过的in_splits？因为如果模型在in_splits（训练集）上的准确率只有30%，而在out_splits（验证集）上也只有30%，你就知道：这不是泛化的问题，是模型根本没学会，或者训练还没结束。
# 如果你在eval_loaders里删掉了in_splits，你的实验记录里就只剩下一堆验证集的分数。
# 当你的实验效果很差时，你会不明白：是模型过拟合太严重了吗？（不知道，因为没测训练集分数）。是模型欠拟合根本没动吗？（不知道，因为没测训练集分数）。 所以，把它们全加起来是为了在评估阶段，给模型做一次“全身体检”。
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
# 如果没有这行代码，你的日志可能长这样：{"step": 100, "acc": [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]} 你完全无法分辨0.4到底是训练集的准确率，还是测试集的。
# 有了这行代码，日志会变成这样：{"step": 100, "env0_in": 0.9, "env1_in": 0.8, "env2_out": 0.4 ...} 这样你一眼就能看出：模型在训练集（env0_in）跑得很好，但在测试集（env2_out）跑得很烂。
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
# 算法类实例化。
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

# 我们有多个训练环境（比如Env_0是照片，Env_1是卡通）。zip(*...)把多个独立的加载器捆绑在一起。当你调用next()时，它会同时从Env_0拿一个Batch，从Env_1拿一个Batch……以此类推。
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
# 这是一个特殊的字典。当你往里面存一个不存在的键时，它会自动创建一个空列表[]。用来存放每一个Checkpoint（检查点）的各项指标。比如记录每100步的Loss、各个环境的Accuracy等。
    checkpoint_vals = collections.defaultdict(lambda: [])

# 在DomainBed中，由于不同环境的数据量不一样（有的多，有的少），“一轮（Epoch）”的定义变得模糊。这里采取了保守策略,取所有训练环境中样本量最小的那一个。
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

# n_steps：总共要跑多少步（比如 5000 步）。如果用户没在命令行指定，就用数据集默认的值。
# checkpoint_freq：每隔多少步保存一次模型并进行全量评估。
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
# 如果你在运行实验时加了--skip_model_save参数，这个函数会直接返回，什么都不存。
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
# 利用Python的序列化机制（Pickle），将上面那个复杂的字典转化成二进制文件,并保存到指定路径上。注意，这段代码是位于save_checkpoint内的。
        torch.save(save_dict, os.path.join(args.output_dir, filename))

# 这个变量用于打印表头，如果是第一次允许，表头为None，即last_results_keys = None，则打印表头，例如['step', 'loss', 'acc']，并把['step', 'loss', 'acc']存入last_results_keys,在第二次验证是，会检查这个变量，如果这个变量不为空，则不打印表头。
    last_results_keys = None
# 为什么这里要用range(start_step, n_steps)，不直接用n_steps？为了支持断点续训。
# 在实际的科研或工业训练中，由于服务器停电、显卡任务被顶替、或者设置了最长运行时间，程序很可能在跑到一半（比如第 500 步）时突然断掉。如果代码写死成 range(n_steps)，那么每次重新启动，模型都会从第 0 步开始重练，之
    for step in range(start_step, n_steps):
        step_start_time = time.time()
# 这是个列表推导。通过.to(device)把图片和标签从内存搬到GPU显存，并将数据存到minibatches_device里。
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
# # 如果是UDA任务，还要额外抓取无标签数据。
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
# 调用算法的更新函数。这里面包含了前向传播、计算 Loss 和反向传播（更新参数）。
        step_vals = algorithm.update(minibatches_device, uda_device)
# 记录这步花了多久。
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
# 记录算法返回的各种指标。
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
# 每隔checkpoint_freq步，模型就会停下来进行一次验证。
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
# 计算距离上次step这段时间里各种指标的的平均值。
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
# 遍历我们之前准备好的“全维度考场” (in, out, uda)。
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
# 看看显存情况。
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
# 结果展示与存档。
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
# 清空这一阶段的记录，准备迎接下一个周期的训练。
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
# 训练全部结束，存下最终模型。
    save_checkpoint('model.pkl')
# 在输出目录创建一个叫'done'的空文件。这样sweep.py扫描时看到这个文件，就知道该任务已完成，不会再重复启动。
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
