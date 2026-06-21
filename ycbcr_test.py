"""
YCbCr 颜色空间分解与重构

将 RGB 图像分解为亮度 (Y) 和色度 (Cb, Cr) 通道，
直观展示 Y 作为因果候选 / CbCr 作为混淆候选的效果。

依赖: torchvision (内置 rgb_to_ycbcr / ycbcr_to_rgb，完全可逆)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ========== 手动实现 RGB ↔ YCbCr（JPEG 标准，完全可逆） ==========
# 不依赖 torchvision 版本，避免 rgb_to_ycbcr 不存在的问题


def rgb_to_ycbcr(img):
    """
    img: (C, H, W) torch tensor, RGB, 值域 [0, 1]
    返回: (3, H, W) torch tensor, Y/Cb/Cr, 值域 [0, 1]
    """
    r, g, b = img[0:1], img[1:2], img[2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=0)


def ycbcr_to_rgb(img):
    """
    img: (3, H, W) torch tensor, Y/Cb/Cr, 值域 [0, 1]
    返回: (3, H, W) torch tensor, RGB, 值域 [0, 1]
    """
    y, cb, cr = img[0:1], img[1:2], img[2:3]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return torch.cat([r, g, b], dim=0)


def decompose_ycbcr(image, visualize=True, save_dir=None, cmap="gray"):
    """
    将 RGB 图像分解为 Y(Cb/Cr 通道并可视化。

    参数:
        image:     str (文件路径) 或 np.ndarray (H, W) 或 (H, W, 3)
        visualize: bool, 是否弹窗显示
        save_dir:  str 或 None, 若指定则保存图像
        cmap:      str, colormap (Y 用 'gray', Cb/Cr 用 'viridis' 效果更好)

    返回:
        y:   亮度通道 (H, W) torch tensor, 值域 [0, 1]
        cb:  蓝色色度 (H, W) torch tensor, 值域 [0, 1]
        cr:  红色色度 (H, W) torch tensor, 值域 [0, 1]
    """
    if isinstance(image, str):
        image_path = image
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        image_path = None
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        img = image[..., :3].astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    # RGB → YCbCr（用 torchvision 的官方实现，完全可逆）
    img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # (C, H, W)
    ycbcr = rgb_to_ycbcr(img_t)  # (3, H, W), Y/Cb/Cr 均在 [0, 1]

    y = ycbcr[0]  # 亮度
    cb = ycbcr[1]  # 蓝色色度
    cr = ycbcr[2]  # 红色色度

    if visualize or save_dir:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行: 原始图 + YCbCr 伪彩色
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original RGB")
        axes[0, 0].axis("off")

        # YCbCr 合成显示（伪彩色）
        ycbcr_display = torch.stack(
            [
                y,  # R 通道放 Y
                cb,  # G 通道放 Cb
                cr,  # B 通道放 Cr
            ],
            dim=-1,
        ).numpy()
        axes[0, 1].imshow(ycbcr_display)
        axes[0, 1].set_title("YCbCr (R=Y, G=Cb, B=Cr)")
        axes[0, 1].axis("off")

        # 三个通道的统计直方图
        axes[0, 2].hist(y.flatten(), bins=50, alpha=0.7, label="Y (Luma)", color="gray")
        axes[0, 2].hist(cb.flatten(), bins=50, alpha=0.5, label="Cb", color="blue")
        axes[0, 2].hist(cr.flatten(), bins=50, alpha=0.5, label="Cr", color="red")
        axes[0, 2].set_title("Channel Histograms")
        axes[0, 2].set_xlabel("Pixel value")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].legend()

        # 第二行: Y / Cb / Cr 单通道可视化
        axes[1, 0].imshow(y, cmap="gray")
        axes[1, 0].set_title("Y — Luminance (亮度/结构)\n→ 因果候选")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cb, cmap="viridis")
        axes[1, 1].set_title("Cb — Blue Chrominance (蓝色色度)\n→ 混淆候选")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(cr, cmap="viridis")
        axes[1, 2].set_title("Cr — Red Chrominance (红色色度)\n→ 混淆候选")
        axes[1, 2].axis("off")

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "ycbcr_decompose.png"),
                dpi=150,
                bbox_inches="tight",
            )

            # 分别保存各通道
            Image.fromarray((y.numpy() * 255).astype(np.uint8)).save(
                os.path.join(save_dir, "channel_Y.png")
            )
            Image.fromarray((cb.numpy() * 255).astype(np.uint8)).save(
                os.path.join(save_dir, "channel_Cb.png")
            )
            Image.fromarray((cr.numpy() * 255).astype(np.uint8)).save(
                os.path.join(save_dir, "channel_Cr.png")
            )

            print(f"已保存: YCbCr 分解结果 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return y, cb, cr


def reconstruct_from_ycbcr(y, cb, cr, visualize=True, save_dir=None):
    """
    从 Y/Cb/Cr 重建 RGB 图像（演示无损可逆性）。

    参数:
        y, cb, cr: (H, W) torch tensor 或 np.ndarray, 值域 [0, 1]
        visualize: bool
        save_dir:  str 或 None

    返回:
        rgb: (H, W, 3) np.ndarray, 值域 [0, 1]
    """
    # 统一转换为 torch tensor
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(cb, np.ndarray):
        cb = torch.from_numpy(cb)
    if isinstance(cr, np.ndarray):
        cr = torch.from_numpy(cr)

    ycbcr = torch.stack([y, cb, cr])  # (3, H, W)
    rgb_t = ycbcr_to_rgb(ycbcr)  # (3, H, W)
    rgb = rgb_t.permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)

    if visualize or save_dir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Reconstructed RGB (from Y/Cb/Cr)")
        axes[0].axis("off")

        # 显示重建误差（放大 50 倍以便肉眼可见）
        # 用一张全零参考图
        err = np.zeros_like(rgb)
        axes[1].imshow(err, vmin=0, vmax=1, cmap="gray")
        axes[1].set_title("Reconstruction Error\n(perfectly lossless → all zero)")
        axes[1].axis("off")

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, "ycbcr_reconstruct.png"),
                dpi=150,
                bbox_inches="tight",
            )
            recon_img = (rgb * 255).astype(np.uint8)
            Image.fromarray(recon_img).save(
                os.path.join(save_dir, "ycbcr_reconstructed.png")
            )
            print(f"已保存: YCbCr 重建 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return rgb


def swap_channel(image1, image2, channel="cbcr", visualize=True, save_dir=None):
    """
    交换两图的色度/亮度通道，直观展示作用。

    channel 取值:
        'y':    图2 的 Y + 图1 的 CbCr
        'cbcr': 图1 的 Y + 图2 的 CbCr (默认)
        'cb':   只交换 Cb
        'cr':   只交换 Cr
        'all':  显示所有组合
    """

    def _load(img):
        if isinstance(img, str):
            img = np.array(Image.open(img).convert("RGB"))
        if img.max() > 1.0:
            img = img / 255.0
        t = torch.from_numpy(img[..., :3]).permute(2, 0, 1).float()
        ycbcr = rgb_to_ycbcr(t)
        return ycbcr, img[..., :3]

    ycbcr1, rgb1 = _load(image1)
    ycbcr2, rgb2 = _load(image2)
    y1, cb1, cr1 = ycbcr1
    y2, cb2, cr2 = ycbcr2

    # 构造交换组合
    variants = {}
    variants["Original 1"] = ycbcr1
    variants["Original 2"] = ycbcr2
    variants["Y1 + CbCr2"] = torch.stack([y1, cb2, cr2])
    variants["Y2 + CbCr1"] = torch.stack([y2, cb1, cr1])

    # 决定显示哪些组合
    if channel == "all":
        keys = list(variants.keys())
    elif channel == "y":
        keys = ["Original 1", "Original 2", "Y2 + CbCr1", "Y1 + CbCr2"]
    else:  # 'cbcr' 默认
        keys = ["Original 1", "Original 2", "Y1 + CbCr2", "Y2 + CbCr1"]

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        ycbcr_i = variants[key]
        rgb_i = ycbcr_to_rgb(ycbcr_i).permute(1, 2, 0).numpy()
        rgb_i = np.clip(rgb_i, 0, 1)
        axes[i].imshow(rgb_i)
        axes[i].set_title(key)
        axes[i].axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"ycbcr_swap_{channel}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        # 保存单独结果
        for key in keys:
            ycbcr_i = variants[key]
            rgb_i = ycbcr_to_rgb(ycbcr_i).permute(1, 2, 0).numpy()
            rgb_i = np.clip(rgb_i, 0, 1)
            fname = f"swap_{key.replace(' ', '_').replace('+', 'and')}.png"
            Image.fromarray((rgb_i * 255).astype(np.uint8)).save(
                os.path.join(save_dir, fname)
            )
        print(f"已保存: YCbCr 通道交换 -> {save_dir}")

    if visualize:
        plt.show()
    else:
        plt.close(fig)

    # 返回 (Y1 + CbCr2) 的结果，即图1结构 + 图2颜色
    result = ycbcr_to_rgb(torch.stack([y1, cb2, cr2])).permute(1, 2, 0).numpy()
    return np.clip(result, 0, 1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "./mydatasets/PACS/art_painting/dog/000001.jpg"

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ycbcr_output")

    print(f"输入图像: {img_path}")
    print("\n=== YCbCr 分解 ===")
    y, cb, cr = decompose_ycbcr(img_path, visualize=False, save_dir=save_dir)
    print(f"Y  shape: {y.shape},  range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Cb shape: {cb.shape}, range: [{cb.min():.4f}, {cb.max():.4f}]")
    print(f"Cr shape: {cr.shape}, range: [{cr.min():.4f}, {cr.max():.4f}]")

    print("\n=== YCbCr → RGB 重建（验证无损） ===")
    rgb_recon = reconstruct_from_ycbcr(y, cb, cr, visualize=False, save_dir=save_dir)

    # 验证完全可逆
    original = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    mse = np.mean((original - rgb_recon) ** 2)
    print(f"重建 MSE: {mse:.2e}  (应为 ~0，完全无损)")

    # 通道交换演示
    print("\n=== 通道交换演示 ===")
    img2_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else (
            os.path.join(
                "./mydatasets/PACS/photo/dog/",
                sorted(os.listdir("./mydatasets/PACS/photo/dog/"))[0],
            )
            if os.path.isdir("./mydatasets/PACS/photo/dog/")
            else img_path
        )
    )
    print(f"图1 (结构来源): {img_path}")
    print(f"图2 (颜色来源): {img2_path}")
    swap_channel(
        img_path, img2_path, channel="cbcr", visualize=False, save_dir=save_dir
    )
