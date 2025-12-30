# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}
# 参数random_val_fn是一个回调函数，回调函数就是把一个函数当作参数，传给另一个函数，被传的函数叫做回调函数。
# 回调的目的1：提供“插件式”的定制能力，也就是说，主函数负责整个流程，回调函数负责具体细节。
# 以_hparam函数为例，_hparam负责整体流程，将超参数注册到hparams字典中，而在主函数关于随机数的部分，不同的超参数需要的处理是不同的，所以通过回调函数，让用户想要什么处理就自己插入什么处理，否则的话，就要写一个很长的ifelse链，判断用户的需要哪个函数处理随机数。
# 回调的目的2：增强代码的复用性。目的1的例子也展现了这一点，即我可以根据我的需求，传入不同的回调函数，让主函数更加通用。
# 回调的目的3：延迟执行。我虽然现在就把函数传给了你，但我不准你立刻跑，你得等到“万事俱备”或者“时机成熟”的时候才能跑。即主程序的要传入回调函数的数据准备好，才能继续跑。
# 简单来说，回调就是：1、把一个函数当作参数，传给另一个函数。2、另一个函数在其内部调用被传的函数。
    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
# 这段代码起到检查的作用，确保同一个参数名不会被定义两次。
        assert(name not in hparams)
# 在计算机里，并没有真正的“随机”。所有的随机数都是通过一个函数算出来的。随机数种子是函数的参数，如果你给这个函数同一个参数，它算出来的数字序列永远是一模一样的。所说的函数，就是随机数生成器。
# misc.seed_hash(random_seed, name）将函数_hparams内的“全局种子”random_seed和当前参数名name混合在一起，生成一个独一无二的“小种子”，并使用这个小种子创建一个专属的随机数生成器。
# random_val_fn(random_state)就是使用一个回调函数random_val_fn，去处理随机数生成器生成的随机数。
# 为什么给每个参数一个专属的随机数生成器（专属来源于专属的随机数种子）？如果共用一个随机数生成器，那么1、先给learning_rate抽数：打印机吐出第 1 个数。2、再给batch_size抽数：打印机吐出第 2 个数。
# 如何你明天把超参数的注册顺序改了，加入删除了learning_rate，那么batch_size就抽出来第一个数了，会造成实验结果不可复现。
# 为什么每个超参数要带一个随机数？深度学习模型对超参数（如学习率lr）非常敏感。如果我们只用默认值（比如 lr=0.01），可能这个值并不是最好的，模型表现一般。如果我们想找到更好的值，最笨的方法是人手一个一个去试，但这太慢了。
# 解决方法：我们给每个参数设定一个范围（比如0.001到0.1），然后让电脑在这个范围内随机抽奖。以前科学家喜欢用“网格搜索”（比如lr选[0.1, 0.01, 0.001]），但后来发现，随机抽样更容易抓到那个最优的点。
# 对于专属随机数生成器，还能解决另一个问题，例如，有DANN和ERM两个模型，为了对比两个模型，我们给DANN和ERM各随机生成20组超参数，跑20次实验，看最好的结果。
# 使用共用随机数生成器会存在那么一个问题：如果DANN的随机参数很好，而ERM的随机参数很差，那对比就不公平了。使用专属随机数生成器，就可以保证，只要全局种子一样，无论你在哪个电脑上跑，无论你跑多少次，这20组“随机抽出来”的参数组合是完全一致且公平的。
# 以学习率为例，如果开启了domainbed随机搜索模式，那么就会同时创建20（以20个为例）个训练任务，每个训练任务有一个随机的学习率，然后取效果好的那个。（是不是domainbed真的是这样待验证）
# 这样不会拖慢训练速度吗？虽然同时训练20个模型看起来很慢，但如果你手动调参：你先试一个0.01，发现效果不好。改代码，再试一个0.005，还是不好。这样反复折腾，可能花费你几天的时间。
# 自动化随机搜索（Random Search）的逻辑是：既然我不知道哪个好，我就利用服务器的多卡并行能力（比如8张显卡同时跑），一次性把这20个坑位都占满。虽然总计算量大了，但你在电脑前等待最终结果的“墙钟时间（Wall-clock time）”大大缩短了。
# 也可以用earlystopping去提前结束效果不好的训练任务，提升效率。
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet50_augmix', True, lambda r: True)
    _hparam('dinov2', False, lambda r: False)
    _hparam('vit', False, lambda r: False)
    _hparam('vit_attn_tune', False, lambda r: False)
    _hparam('freeze_bn', False, lambda r: False)
    _hparam('lars', False, lambda r: False)
    _hparam('linear_steps', 500, lambda r: 500)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('vit_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
# 为什么不直接从均匀分布里抽数，还得用随机数生成器，根据随机数生成器生成的值抽均匀分布？
# 直接随机抽，无法保证代码的可复现性。靠随机数生成器抽，可以保证实验结果可浮现。另外，抽数，必须靠随机数生成器，即在随机数生成器上调用函数来抽某一分布的数。没有随机数生成器，计算机是无法自己直接抽某个分布的数的。
# 计算机本质上是一个确定性机器。如果你不给它特殊的指令，它执行1+1永远等于2。它自己是不会“掷骰子”的。所以要靠随机数生成器生成“随机数”，然后以从均匀分布抽数为例。
# r.uniform会告诉生成器：“请把产生的数，平均地铺在A到B这个区间里。然后抽出来一个给我。
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RDM": 
        if dataset in ['DomainNet']: 
            _hparam('rdm_lambda', 0.5, lambda r: r.uniform(0.1, 1.0))
        elif dataset in ['PACS', 'TerraIncognita']:
            _hparam('rdm_lambda', 5.0, lambda r: r.uniform(1.0, 10.0))
        else:
            _hparam('rdm_lambda', 5.0, lambda r: r.uniform(0.1, 10.0))
            
        if dataset == 'DomainNet':
            _hparam('rdm_penalty_anneal_iters', 2400, lambda r: int(r.uniform(1500, 3000)))
        else:
            _hparam('rdm_penalty_anneal_iters', 1500, lambda r: int(r.uniform(800, 2700)))
            
        if dataset in ['TerraIncognita', 'OfficeHome', 'DomainNet']:
            _hparam('variance_weight', 0.0, lambda r: r.choice([0.0]))
        else:
            _hparam('variance_weight', 0.004, lambda r: r.uniform(0.001, 0.007))
            
        _hparam('rdm_lr', 1.5e-5, lambda r: r.uniform(8e-6, 2e-5))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  r.choice([1, 2]))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    elif algorithm == 'EQRM':
        _hparam('eqrm_quantile', 0.75, lambda r: r.uniform(0.5, 0.99))
        _hparam('eqrm_burnin_iters', 2500, lambda r: 10 ** r.uniform(2.5, 3.5))
        _hparam('eqrm_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))

    elif algorithm == 'ERMPlusPlus':
        _hparam('linear_lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    elif algorithm == 'URM':
        _hparam('urm', 'adversarial', lambda r: str(r.choice(['adversarial']))) # 'adversarial'
        
        _hparam('urm_adv_lambda', 0.1, lambda r: float(r.uniform(0,0.2)))
        _hparam('urm_discriminator_label_smoothing', 0, lambda r: float(r.uniform(0, 0)))
        _hparam('urm_discriminator_optimizer', 'adam', lambda r: str(r.choice(['adam'])))
        _hparam('urm_discriminator_hidden_layers', 1, lambda r: int(r.choice([1,2,3])))
        _hparam('urm_generator_output', 'tanh', lambda r: str(r.choice(['tanh', 'relu'])))
                
        if dataset in SMALL_IMAGES:
            _hparam('urm_discriminator_lr', 1e-3, lambda r: 10**r.uniform(-5.5, -3.5))
        else:
            _hparam('urm_discriminator_lr', 5e-5, lambda r: 10**r.uniform(-6, -4.5))


    if algorithm == "ADRMX":
        _hparam('cnt_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('dclf_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('disc_lambda', 0.75, lambda r: r.choice([0.75]))
        _hparam('rmxd_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('d_steps_per_g_step', 2, lambda r: r.choice([2]))
        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        _hparam('mlp_width', 256, lambda r: r.choice([256]))
        _hparam('mlp_depth', 9, lambda r: int(r.choice([8, 9, 10])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0]))


    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        if algorithm == "ADRMX":
            _hparam('lr', 3e-3, lambda r: r.choice([5e-4, 1e-3, 2e-3, 3e-3]))
        else:
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        if algorithm == "ADRMX":
            _hparam('lr', 3e-5, lambda r: r.choice([2e-5, 3e-5, 4e-5, 5e-5]))
        else:
            _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif algorithm == 'RDM':
        if dataset in ['DomainNet', 'TerraIncognita']:
            _hparam('batch_size', 40, lambda r: int(r.uniform(30, 60)))
        else:
            _hparam('batch_size', 88, lambda r: int(r.uniform(70, 100)))
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
