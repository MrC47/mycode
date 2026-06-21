"""
对比实验：FFT-mix vs DWT-mix vs Baseline

FFT-mix:  保持相位不变，幅度谱线性插值后重建
DWT-mix:  保持 LL 不变，细节子带 (LH/HL/HH) 线性插值后重建

用法:
    # 默认 alpha=0.5，3 种模式依次运行
    docker container exec -it mycode python experiment_freq_mixing.py

    # 指定插值系数和运行模式
    docker container exec -it mycode python experiment_freq_mixing.py --alpha 0.3 --mode all

    # 只跑 baseline
    docker container exec -it mycode python experiment_freq_mixing.py --mode none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import os
import sys
import random
import time
import argparse
import json

import pywt


# ======================== 频率混合函数 ========================

def denormalize(tensor):
    """ImageNet 逆归一化: 还原到 [0,1] 范围"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def normalize(tensor):
    """ImageNet 归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def fft_mix(img_a, img_b, alpha=0.5):
    """
    FFT 混合: 相位来自 A，幅度谱线性插值。
    输入: img_a, img_b: (C, H, W) torch tensor, 值域 [0,1]
    返回: mixed: (C, H, W), 值域 [0,1]
    """
    f_a = torch.fft.fft2(img_a)
    f_b = torch.fft.fft2(img_b)

    amp_a = torch.abs(f_a)
    amp_b = torch.abs(f_b)
    phase_a = torch.angle(f_a)

    amp_mixed = alpha * amp_a + (1.0 - alpha) * amp_b
    f_mixed = amp_mixed * torch.exp(1j * phase_a)
    mixed = torch.fft.ifft2(f_mixed).real
    return mixed.clamp(0, 1)


def compute_z(detail_channels, approx_channel):
    """
    计算风格浓度 Z: 细节能量 / 总能量。
    输入: 均为 numpy 2D 数组。
    返回: float 标量 Z ∈ [0, 1]。
    """
    cH, cV, cD = detail_channels
    detail_energy = (cH**2 + cV**2 + cD**2).sum()
    approx_energy = (approx_channel**2).sum()
    total_energy = approx_energy + detail_energy
    return float(detail_energy / (total_energy + 1e-8))


def conditional_dwt_mix(img_a, img_b, wavelet='db1',
                        alpha_min=0.3, alpha_max=1.0):
    """
    条件 DWT 混合: Z（风格浓度）→ α，每通道独立计算并取平均 Z。
    Z 高 → α 小（强干预，混入更多 B 的风格），
    Z 低 → α 大（轻干预，保留 A 的风格）。
    """
    C, H, W = img_a.shape
    mixed_channels = []
    z_values = []

    for c in range(C):
        a_np = img_a[c].cpu().numpy()
        b_np = img_b[c].cpu().numpy()

        cA_a, (cH_a, cV_a, cD_a) = pywt.dwt2(a_np, wavelet)
        cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b_np, wavelet)

        # 计算 Z（当前通道的风格浓度）
        Z = compute_z((cH_a, cV_a, cD_a), cA_a)
        z_values.append(Z)

        # Z → α: 线性映射到 [alpha_min, alpha_max]
        # Z=0（平滑）→ α=alpha_max（轻干预）
        # Z=1（纹理密集）→ α=alpha_min（强干预）
        alpha = alpha_max - (alpha_max - alpha_min) * Z

        # LL 来自 A，细节插值
        cA = cA_a
        cH = alpha * cH_a + (1.0 - alpha) * cH_b
        cV = alpha * cV_a + (1.0 - alpha) * cV_b
        cD = alpha * cD_a + (1.0 - alpha) * cD_b

        recon = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
        recon = recon[:H, :W]
        mixed_channels.append(torch.from_numpy(recon).float())

    mixed = torch.stack(mixed_channels)
    return mixed.clamp(0, 1)


def dwt_mix(img_a, img_b, alpha=0.5, wavelet='db1'):
    """
    DWT 混合: LL 来自 A，细节子带线性插值。
    pywt 基于 numpy，所以需要 CPU 转换。
    """
    C, H, W = img_a.shape
    mixed_channels = []

    for c in range(C):
        a_np = img_a[c].cpu().numpy()
        b_np = img_b[c].cpu().numpy()

        cA_a, (cH_a, cV_a, cD_a) = pywt.dwt2(a_np, wavelet)
        cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b_np, wavelet)

        # LL 来自 A，细节插值
        cA = cA_a
        cH = alpha * cH_a + (1.0 - alpha) * cH_b
        cV = alpha * cV_a + (1.0 - alpha) * cV_b
        cD = alpha * cD_a + (1.0 - alpha) * cD_b

        recon = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
        # 裁剪到原始尺寸（idwt2 在奇数尺寸时可能多 1 像素）
        recon = recon[:H, :W]
        mixed_channels.append(torch.from_numpy(recon).float())

    mixed = torch.stack(mixed_channels)
    return mixed.clamp(0, 1)


# ======================== 混合数据集 ========================

class FrequencyMixDataset(Dataset):
    """
    包装域数据集，在线进行频率域混合。

    mode:
        'none' — 原始图像，不做混合
        'fft'  — 相位不变，幅度谱插值
        'dwt'  — LL 不变，细节子带插值
        'cdwt' — 条件 DWT: Z（风格浓度）→ α，每样本自适应
    """

    def __init__(self, env_datasets, mode='fft', alpha=0.5):
        """
        env_datasets: list of ImageFolder (每个域一个)
        mode: 'none', 'fft', 'dwt', 'cdwt'
        alpha: 插值系数 (仅 fft/dwt 模式使用)
        """
        self.env_datasets = env_datasets
        self.mode = mode
        self.alpha = alpha
        self.num_domains = len(env_datasets)

        # 构建扁平索引: (domain_idx, sample_idx)
        self.sample_map = []
        for d_idx, ds in enumerate(env_datasets):
            for s_idx in range(len(ds)):
                self.sample_map.append((d_idx, s_idx))

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        d_a, s_a = self.sample_map[idx]
        img_a, label_a = self.env_datasets[d_a][s_a]

        if self.mode == 'none':
            return img_a, label_a

        # 从下一个域取 B（确定性配对，避免 DataLoader worker 的随机种子问题）
        d_b = (d_a + 1) % self.num_domains
        s_b = idx % len(self.env_datasets[d_b])
        img_b, _ = self.env_datasets[d_b][s_b]

        # 反归一化 → 混合 → 重新归一化
        img_a_raw = denormalize(img_a)
        img_b_raw = denormalize(img_b)

        if self.mode == 'fft':
            mixed = fft_mix(img_a_raw, img_b_raw, self.alpha)
        elif self.mode == 'dwt':
            mixed = dwt_mix(img_a_raw, img_b_raw, self.alpha)
        elif self.mode == 'cdwt':
            mixed = conditional_dwt_mix(img_a_raw, img_b_raw)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        mixed = normalize(mixed)
        return mixed, label_a


# ======================== 训练 / 评估 ========================

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(loader)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def get_pacs_loaders(data_dir, test_domain, batch_size=32, num_workers=4):
    """
    手动加载 PACS: 用 torchvision 的 ImageFolder。
    test_domain: 留出的测试域 (如 'art_painting')
    返回: (train_envs, test_loader, num_classes)
    """
    base_dir = os.path.join(data_dir, "PACS")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"PACS 目录不存在: {base_dir}")

    domains = sorted([f.name for f in os.scandir(base_dir) if f.is_dir()])
    if test_domain not in domains:
        raise ValueError(f"域 '{test_domain}' 不在 {domains} 中")

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        torchvision.transforms.RandomGrayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_envs = []
    test_loader = None
    num_classes = None

    for domain in domains:
        path = os.path.join(base_dir, domain)
        if domain == test_domain:
            ds = ImageFolder(path, transform=test_transform)
            test_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)
        else:
            ds = ImageFolder(path, transform=train_transform)
            train_envs.append(ds)

        if num_classes is None:
            num_classes = len(ds.classes)

    return train_envs, test_loader, num_classes


# ======================== 主流程 ========================

def run_experiment(mode, alpha, train_envs, test_loader, num_classes,
                   device, epochs=10, lr=0.001, batch_size=32, num_workers=4):
    """运行一次实验，返回 {test_domain: accuracy}"""
    tag = {'none': 'Baseline', 'fft': 'FFT-mix', 'dwt': 'DWT-mix', 'cdwt': 'Cond-DWT'}[mode]
    print(f"\n{'='*60}")
    print(f"  [{tag}] mode={mode}, alpha={alpha}")
    print(f"{'='*60}")

    # 模型
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(2048, num_classes)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 数据集
    train_dataset = FrequencyMixDataset(train_envs, mode=mode, alpha=alpha)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    # 训练
    best_acc = 0.0
    start = time.time()
    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f"  Epoch {epoch+1:2d}/{epochs} | loss={loss:.4f} | train_acc={acc:.4f}", end='')

        # 每 5 个 epoch 评估一次
        if (epoch + 1) % 5 == 0:
            test_acc = evaluate(model, test_loader, device)
            best_acc = max(best_acc, test_acc)
            print(f" | test_acc={test_acc:.4f}", end='')
        print()

    elapsed = time.time() - start

    # 最终评估
    final_acc = evaluate(model, test_loader, device)
    print(f"  [{tag}] Final test_acc={final_acc:.4f} | Best={best_acc:.4f} | Time={elapsed:.0f}s")

    return {
        'mode': mode,
        'alpha': alpha,
        'test_acc': final_acc,
        'best_test_acc': best_acc,
        'epochs': epochs,
        'time_sec': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='频率混合对比实验')
    parser.add_argument('--data_dir', default='./mydatasets',
                        help='数据集根目录')
    parser.add_argument('--dataset', default='PACS',
                        choices=['PACS'], help='数据集')
    parser.add_argument('--test_domain', default='art_painting',
                        help='留出的测试域')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='插值系数 (0.0 = 完全用 B, 1.0 = 完全用 A)')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'none', 'fft', 'dwt', 'cdwt'],
                        help='运行模式')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"数据集: {args.dataset}, 测试域: {args.test_domain}, alpha={args.alpha}")

    # 加载数据
    train_envs, test_loader, num_classes = get_pacs_loaders(
        args.data_dir, args.test_domain, args.batch_size)
    print(f"训练域数: {len(train_envs)}, 类别数: {num_classes}")

    # 确定要运行的模式
    modes = ['none', 'fft', 'dwt'] if args.mode == 'all' else [args.mode]

    # 依次运行
    results = []
    for mode in modes:
        result = run_experiment(
            mode=mode, alpha=args.alpha,
            train_envs=train_envs, test_loader=test_loader,
            num_classes=num_classes, device=device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
        results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  结果汇总 (test_domain={args.test_domain}, alpha={args.alpha})")
    print(f"{'='*60}")
    for r in results:
        tag = {'none': 'Baseline', 'fft': 'FFT-mix', 'dwt': 'DWT-mix', 'cdwt': 'Cond-DWT'}[r['mode']]
        print(f"  {tag:15s} | test_acc={r['test_acc']:.4f} | best={r['best_test_acc']:.4f} | time={r['time_sec']:.0f}s")


if __name__ == '__main__':
    main()
