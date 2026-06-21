"""
DCT（离散余弦变换）分解与重构

DCT 是 JPEG 压缩的核心：将图像从空间域变换到频率域，
能量高度集中在低频（左上角），丢弃高频可以实现无损压缩。

整图 DCT  vs  8×8 分块 DCT：
  - 整图 DCT：全局频率分析，适合理解频率分布
  - 分块 DCT：局部频率分析，JPEG 标准方式
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# ========== DCT 手动实现（不依赖 scipy） ==========

def dct_2d(x):
    """
    2D DCT（Type-II，正交归一化），直接矩阵乘法实现。
    x: (H, W) numpy array
    返回: (H, W) DCT 系数
    """
    H, W = x.shape
    # 构造 DCT 矩阵
    def _dct_matrix(N):
        n = np.arange(N)
        k = np.arange(N)[:, None]
        mat = np.cos(np.pi / N * (n + 0.5) * k)
        mat[0] *= 1.0 / np.sqrt(2)  # 正交归一化
        return mat * np.sqrt(2.0 / N)

    C_h = _dct_matrix(H)
    C_w = _dct_matrix(W)
    return C_h @ x @ C_w.T


def idct_2d(y):
    """
    2D IDCT（Type-III，dct_2d 的逆变换）。
    y: (H, W) DCT 系数
    返回: (H, W) 重建信号
    """
    H, W = y.shape
    def _idct_matrix(N):
        n = np.arange(N)[:, None]
        k = np.arange(N)
        mat = np.cos(np.pi / N * (n + 0.5) * k)
        mat[:, 0] *= 1.0 / np.sqrt(2)
        return mat * np.sqrt(2.0 / N)

    C_h = _idct_matrix(H)
    C_w = _idct_matrix(W)
    return C_h @ y @ C_w.T


# ========== 核心函数 ==========

def decompose_dct(image, visualize=True, save_dir=None, cmap='inferno'):
    """
    对图像进行全图 DCT 分解，可视化频谱。

    参数:
        image: str 路径 或 np.ndarray
        visualize, save_dir: 同前
        cmap: DCT 频谱的 colormap

    返回:
        dct_coeffs: (C, H, W) DCT 系数（每个通道独立）
    """
    if isinstance(image, str):
        img = np.array(Image.open(image).convert('RGB')).astype(np.float32)
    else:
        img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    is_gray = (img.ndim == 2)
    if is_gray:
        img = img[..., np.newaxis]

    H, W, C = img.shape
    dct_coeffs = np.zeros_like(img)

    for c in range(C):
        dct_coeffs[:, :, c] = dct_2d(img[:, :, c])

    if is_gray:
        dct_coeffs = dct_coeffs[:, :, 0]
        img = img[:, :, 0]

    if visualize or save_dir:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # 原图
        axes[0, 0].imshow(img.squeeze(), cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # DCT 频谱（log 尺度）
        disp = np.log1p(np.abs(dct_coeffs))
        axes[0, 1].imshow(disp.squeeze(), cmap=cmap)
        axes[0, 1].set_title('DCT Spectrum (log scale)\nDC=左上角, 高频=右下角')
        axes[0, 1].axis('off')

        # 能量集中度：按频率累计
        coeffs_sq = dct_coeffs_sq(dct_coeffs) if dct_coeffs.ndim == 2 else (dct_coeffs**2).sum(axis=-1)
        total_e = coeffs_sq.sum()
        energy_dist = []
        for frac in np.linspace(0.01, 1.0, 100):
            mask = _low_freq_mask(H, W, frac)
            retained = (coeffs_sq * mask).sum() / total_e
            energy_dist.append(retained)
        axes[0, 2].plot(np.linspace(0.01, 1.0, 100) * 100, energy_dist)
        axes[0, 2].axhline(0.9, color='r', linestyle='--', alpha=0.5, label='90%')
        axes[0, 2].set_xlabel('Coefficients kept (%)')
        axes[0, 2].set_ylabel('Energy retained')
        axes[0, 2].set_title('Energy Concentration')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 第 1 行第 4 列: DCT 系数值分布直方图
        flat_coeffs = dct_coeffs_sq(dct_coeffs).flatten() if dct_coeffs.ndim == 2 else (dct_coeffs**2).sum(axis=-1).flatten()
        axes[0, 3].hist(np.log1p(flat_coeffs + 1e-10), bins=100, color='purple', alpha=0.7)
        axes[0, 3].set_title('DCT Coefficient Distribution (log)')
        axes[0, 3].set_xlabel('log(1+|coeff|)')
        axes[0, 3].set_ylabel('Count')
        axes[0, 3].set_yscale('log')
        axes[0, 3].grid(True, alpha=0.3)

        # 第 2 行: 保留不同比例的重建效果
        fractions = [0.01, 0.05, 0.10, 0.50]
        for i, frac in enumerate(fractions):
            recon = _reconstruct_keep(dct_coeffs, frac)
            recon_disp = recon.squeeze()
            if recon_disp.ndim == 2:
                axes[1, i].imshow(recon_disp, cmap='gray')
            else:
                recon_disp = np.clip(recon_disp, 0, 1)
                axes[1, i].imshow(recon_disp)
            axes[1, i].set_title(f'Keep {frac*100:.0f}% coeffs')
            axes[1, i].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'dct_decompose.png'), dpi=150, bbox_inches='tight')
            print(f"已保存: DCT 分解 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return dct_coeffs


def reconstruct_dct(dct_coeffs, keep_fraction=0.5, is_gray=False, visualize=True, save_dir=None):
    """
    从 DCT 系数重建，可选择保留多少比例的低频系数。

    参数:
        dct_coeffs:     DCT 系数 (H, W) 或 (H, W, C)
        keep_fraction:  保留的低频比例 (0~1)
        is_gray:        是否为灰度图
        visualize:      是否可视化
        save_dir:       保存目录

    返回:
        reconstructed: 重建图像, [0, 1]
    """
    # 生成低频掩码并应用
    if dct_coeffs.ndim == 2:
        h, w = dct_coeffs.shape
        mask = _low_freq_mask(h, w, keep_fraction)
        masked = dct_coeffs * mask
        recon = idct_2d(masked)
    else:
        h, w, c = dct_coeffs.shape
        mask = _low_freq_mask(h, w, keep_fraction)
        channels = []
        for ch in range(c):
            channels.append(idct_2d(dct_coeffs[:, :, ch] * mask))
        recon = np.stack(channels, axis=-1)

    recon = np.clip(recon, 0, 1)

    # 低通/高通分离演示
    if visualize or save_dir:
        # 高通 = 原 - 低通
        if dct_coeffs.ndim == 2:
            high_mask = 1.0 - _low_freq_mask(h, w, keep_fraction)
            high = idct_2d(dct_coeffs * high_mask)
        else:
            high_channels = []
            for ch in range(c):
                high_channels.append(idct_2d(dct_coeffs[:, :, ch] * (1.0 - _low_freq_mask(h, w, keep_fraction))))
            high = np.stack(high_channels, axis=-1)
        high = np.clip(high, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(recon.squeeze(), cmap='gray')
        axes[0].set_title(f'Low-pass (keep {keep_fraction*100:.0f}%)')
        axes[0].axis('off')
        axes[1].imshow(high.squeeze(), cmap='gray')
        axes[1].set_title(f'High-pass (丢掉的部分)')
        axes[1].axis('off')
        # 原图差值
        # 无法拿到原图时展示 0~1 映射
        diff_disp = (high.squeeze() - high.squeeze().min()) / max(high.squeeze().max() - high.squeeze().min(), 1e-8)
        axes[2].imshow(diff_disp, cmap='gray')
        axes[2].set_title('High-pass (normalized)')
        axes[2].axis('off')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'dct_reconstruct.png'), dpi=150, bbox_inches='tight')
            recon_img = (recon * 255).astype(np.uint8)
            Image.fromarray(recon_img).save(os.path.join(save_dir, 'dct_reconstructed.png'))
            print(f"已保存: DCT 重建 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return recon


def block_dct_demo(image, block_size=8, visualize=True, save_dir=None):
    """
    8×8 分块 DCT 演示（JPEG 标准方式）。
    展示每个块的 DC（均值）和 AC（高频）分离效果。
    """
    if isinstance(image, str):
        img = np.array(Image.open(image).convert('RGB')).astype(np.float32)
    else:
        img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    H, W, C = img.shape
    # 裁剪到 block_size 的整数倍
    H, W = H - H % block_size, W - W % block_size
    img = img[:H, :W]

    # 逐块 DCT
    dct_blocks = np.zeros_like(img)
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            for c in range(C):
                block = img[y:y+block_size, x:x+block_size, c]
                dct_blocks[y:y+block_size, x:x+block_size, c] = dct_2d(block)

    # DC-only 重建（只保留每个块的 DC 系数）
    dc_only = np.zeros_like(img)
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            for c in range(C):
                block_coeff = dct_blocks[y:y+block_size, x:x+block_size, c]
                # 保留 DC，清空 AC
                ac_only_block = np.zeros_like(block_coeff)
                ac_only_block[0, 0] = block_coeff[0, 0]  # 只保留 DC
                dc_only[y:y+block_size, x:x+block_size, c] = idct_2d(ac_only_block)

    # AC-only 重建（清空 DC，保留 AC）
    ac_only = np.zeros_like(img)
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            for c in range(C):
                block_coeff = dct_blocks[y:y+block_size, x:x+block_size, c]
                ac_only_block = block_coeff.copy()
                ac_only_block[0, 0] = 0  # 去掉 DC
                ac_only[y:y+block_size, x:x+block_size, c] = idct_2d(ac_only_block)
    # AC-only 往往是[-0.5, 0.5]范围的零均值信号，映射到[0,1]显示
    ac_only_display = ac_only - ac_only.min()
    ac_only_display = ac_only_display / max(ac_only_display.max(), 1e-8)

    if visualize or save_dir:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(img)
        axes[0].set_title(f'Original ({H}×{W})')
        axes[0].axis('off')

        # DCT 系数（第 1 通道）
        dct_log = np.log1p(np.abs(dct_blocks[:, :, 0]))
        axes[1].imshow(dct_log, cmap='inferno')
        axes[1].set_title(f'{block_size}×{block_size} Block DCT')
        axes[1].axis('off')

        axes[2].imshow(dc_only)
        axes[2].set_title('DC Only (每块只保留均值)\n块状伪影明显的平滑版本')
        axes[2].axis('off')

        axes[3].imshow(ac_only_display)
        axes[3].set_title('AC Only (去掉每个块的DC)\n只有块内纹理/边缘')
        axes[3].axis('off')

        # 单个块的放大显示
        block_y, block_x = H // 4, W // 4  # 取中部一个块
        zoom_coeff = dct_blocks[block_y:block_y+block_size, block_x:block_x+block_size, 0]
        ax4 = axes[4]
        im = ax4.imshow(zoom_coeff, cmap='coolwarm', vmin=-np.abs(zoom_coeff).max(), vmax=np.abs(zoom_coeff).max())
        for i in range(block_size):
            for j in range(block_size):
                val = zoom_coeff[i, j]
                color = 'white' if abs(val) < np.abs(zoom_coeff).max() * 0.3 else 'black'
                ax4.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)
        ax4.set_title(f'Single {block_size}×{block_size} Block DCT\n(DC=左上, 右下=最高频)')
        ax4.axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'dct_block_demo.png'), dpi=150, bbox_inches='tight')
            Image.fromarray((dc_only * 255).astype(np.uint8)).save(os.path.join(save_dir, 'dct_dc_only.png'))
            print(f"已保存: DCT 分块演示 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return dct_blocks, dc_only, ac_only


# ========== 辅助函数 ==========

def _low_freq_mask(h, w, keep_fraction):
    """
    生成低频保留掩码：保留左上角 keep_fraction 比例的系数。
    用椭圆区域模拟频率截止。
    """
    cy, cx = 0, 0  # DC 在 (0, 0)
    radius_y = int(np.sqrt(keep_fraction) * h)
    radius_x = int(np.sqrt(keep_fraction) * w)
    radius_y = max(min(radius_y, h), 1)
    radius_x = max(min(radius_x, w), 1)

    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) / radius_y) ** 2 + ((xx - cx) / radius_x) ** 2 <= 1
    return mask.astype(float)


def dct_coeffs_sq(coeffs):
    """计算能量分布（系数平方）。"""
    if coeffs.ndim == 3:
        return (coeffs ** 2).sum(axis=-1)
    return coeffs ** 2


def _reconstruct_keep(dct_coeffs, keep_fraction):
    """保留 top-k% 低频系数重建。"""
    if dct_coeffs.ndim == 2:
        h, w = dct_coeffs.shape
        mask = _low_freq_mask(h, w, keep_fraction)
        return np.clip(idct_2d(dct_coeffs * mask), 0, 1)
    else:
        h, w, c = dct_coeffs.shape
        mask = _low_freq_mask(h, w, keep_fraction)
        channels = [np.clip(idct_2d(dct_coeffs[:, :, ch] * mask), 0, 1) for ch in range(c)]
        return np.stack(channels, axis=-1)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "./mydatasets/PACS/art_painting/dog/000001.jpg"

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dct_output")

    print(f"输入图像: {img_path}")

    # 1. 全图 DCT
    print("\n=== 全图 DCT 分解 ===")
    dct_coeffs = decompose_dct(img_path, visualize=False, save_dir=save_dir)
    print(f"DCT 系数形状: {dct_coeffs.shape}")

    # 计算能量集中度
    if dct_coeffs.ndim == 3:
        energy_map = (dct_coeffs ** 2).sum(axis=-1)
    else:
        energy_map = dct_coeffs ** 2
    total_energy = energy_map.sum()
    h, w = energy_map.shape

    for frac in [0.01, 0.05, 0.10, 0.25, 0.50]:
        mask = _low_freq_mask(h, w, frac)
        retained = (energy_map * mask).sum() / total_energy * 100
        print(f"  保留 {frac*100:.0f}% 系数 → 能量保留 {retained:.1f}%")

    # 2. DCT 重建
    print("\n=== DCT 频率截断重建 ===")
    for frac in [0.01, 0.05, 0.10, 0.25, 0.50]:
        recon = reconstruct_dct(dct_coeffs, keep_fraction=frac, visualize=False)
        # MSE 用原图
        original = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        orig_h, orig_w = original.shape[:2]
        recon_cropped = recon[:orig_h, :orig_w]
        if recon_cropped.shape != original.shape:
            original = original[:recon_cropped.shape[0], :recon_cropped.shape[1]]
        mse = np.mean((original - recon_cropped) ** 2)
        print(f"  keep={frac*100:.0f}%:  MSE={mse:.4e}")

    # 3. 8×8 分块 DCT 演示
    print("\n=== 8×8 分块 DCT 演示 ===")
    block_dct_demo(img_path, block_size=8, visualize=False, save_dir=save_dir)
    print("完成")
