from calendar import LocaleHTMLCalendar

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt


def decompose_wavelet(image, wavelet='db1', level=1, visualize=True, save_dir=None, cmap='gray'):
    """
    将输入图像进行小波分解，可视化各子带。

    参数:
        image:     str (文件路径) 或 np.ndarray (H, W) 或 (H, W, C)
        wavelet:   str, 小波基 (默认 'db1'=Haar)
        level:     int, 分解层数
        visualize: bool, 是否弹出窗口可视化
        save_dir:  str 或 None, 若指定则保存图像
        cmap:      str, colormap

    返回:
        coeffs: pywt.wavedec2 格式的系数 (可用于重构)
    """
    if isinstance(image, str):
        image_path = image
        img = np.array(Image.open(image_path)).astype(np.float32)
    else:
        image_path = None
        img = image.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    is_gray = (img.ndim == 2)

    # --- 小波分解 ---
    # 对每个通道单独分解
    if is_gray:
        coeffs = pywt.wavedec2(img, wavelet, level=level)
    else:
        # RGB: 对各通道分别做小波分解
        coeffs_c = []
        for c in range(img.shape[2]):
            coeffs_c.append(pywt.wavedec2(img[:, :, c], wavelet, level=level))
        coeffs = coeffs_c

    # --- 可视化 ---
    if visualize or save_dir:
        if is_gray:
            _plot_wavelet_decomposition(img, coeffs, wavelet, level, cmap, visualize, save_dir)
        else:
            # RGB 显示每个通道的第一层子带
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            titles = ['Approx (LL)', 'Horizontal (LH)', 'Vertical (HL)', 'Diagonal (HH)']
            for c in range(3):
                cA = coeffs[c][0]
                cH = coeffs[c][1][0]
                cV = coeffs[c][1][1]
                cD = coeffs[c][1][2]
                for j, subband in enumerate([cA, cH, cV, cD]):
                    axes[c][j].imshow(_norm(subband), cmap=cmap)
                    axes[c][j].set_title(f'Channel {c} - {titles[j]}')
                    axes[c][j].axis('off')
            plt.tight_layout()

            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, 'wavelet_decompose_rgb.png'), dpi=150, bbox_inches='tight')

            if visualize:
                plt.show()
            else:
                plt.close(fig)

    return coeffs


def _apply_weight_to_coeffs(coeffs, ll_weight, lh_weight, hl_weight, hh_weight):
    """对 wavedec2 格式的系数应用子带权重，原地修改。"""
    coeffs[0] = coeffs[0] * ll_weight
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        coeffs[i] = (cH * lh_weight, cV * hl_weight, cD * hh_weight)
    return coeffs


def reconstruct_from_wavelet(coeffs, wavelet='db1', is_gray=True, original_shape=None,
                             ll_weight=1.0, lh_weight=1.0, hl_weight=1.0, hh_weight=1.0,
                             visualize=True, save_dir=None, cmap='gray'):
    """
    从小波系数重建图像。

    参数:
        coeffs:         pywt.wavedec2 的返回值 (或 RGB 的列表)
        wavelet:        str, 小波基 (需与分解时一致)
        is_gray:        bool, 是否为灰度图
        original_shape: tuple 或 None, (H, W) 或 (H, W, C), 若提供则裁剪到该尺寸
        ll_weight:      float, 近似分量 (LL) 权重
        lh_weight:      float, 水平细节 (LH) 权重
        hl_weight:      float, 垂直细节 (HL) 权重
        hh_weight:      float, 对角细节 (HH) 权重
        visualize:      bool, 是否弹出窗口可视化
        save_dir:       str 或 None, 若指定则保存图像
        cmap:           str, colormap

    返回:
        reconstructed: np.ndarray, 重建图像, 值域 [0, 1]
    """
    # 对系数应用子带权重
    if is_gray:
        _apply_weight_to_coeffs(coeffs, ll_weight, lh_weight, hl_weight, hh_weight)
        reconstructed = pywt.waverec2(coeffs, wavelet)
    else:
        rec_channels = []
        for c in range(len(coeffs)):
            _apply_weight_to_coeffs(coeffs[c], ll_weight, lh_weight, hl_weight, hh_weight)
            rec_channels.append(pywt.waverec2(coeffs[c], wavelet))
        reconstructed = np.stack(rec_channels, axis=-1)

    # 裁剪到原始尺寸（waverec2 在奇数尺寸时可能多 1 像素）
    if original_shape is not None:
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]

    reconstructed = np.clip(reconstructed, 0, 1)

    if visualize or save_dir:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(reconstructed.squeeze(), cmap=cmap)
        ax.set_title(f'Reconstructed ({wavelet}, {len(coeffs)} levels)')
        ax.axis('off')
        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'wavelet_reconstructed.png'), dpi=150, bbox_inches='tight')
            recon_img = (reconstructed.squeeze() * 255).astype(np.uint8)
            Image.fromarray(recon_img).save(os.path.join(save_dir, 'wavelet_reconstructed_clean.png'))
            print(f"已保存: 小波重建 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return reconstructed


def fusion_wavelet(image1, image2, wavelet='db1', level=1, visualize=True, save_dir=None, cmap='gray'):
    """
    用图片1的 LL + 图片2的细节 (LH/HL/HH) 融合重建。

    参数:
        image1:    str 或 np.ndarray — 提供 LL 近似分量（整体结构）
        image2:    str 或 np.ndarray — 提供细节分量（纹理/边缘）
        wavelet:   str, 小波基
        level:     int, 分解层数
        visualize: bool, 是否弹出窗口可视化
        save_dir:  str 或 None, 若指定则保存结果
        cmap:      str, colormap

    返回:
        reconstructed: np.ndarray, 融合重建图像, 值域 [0, 1]
    """
    import os

    # 分别分解
    coeffs1 = decompose_wavelet(image1, wavelet=wavelet, level=level, visualize=False)
    coeffs2 = decompose_wavelet(image2, wavelet=wavelet, level=level, visualize=False)

    is_gray = len(coeffs1) == level + 1 and not isinstance(coeffs1[0], list)

    if is_gray:
        # 构造融合系数: LL 来自图1, 细节来自图2
        fused_coeffs = [coeffs1[0].copy()]
        for lvl in range(1, level + 1):
            _, cV2, cD2 = coeffs2[lvl]  # 这里顺序: (cH, cV, cD)
            # 用图2的细节替换
            fused_coeffs.append(coeffs2[lvl])
        reconstructed = pywt.waverec2(fused_coeffs, wavelet)
    else:
        # RGB: 每通道分别融合
        rec_channels = []
        for c in range(len(coeffs1)):
            fused_c = [coeffs1[c][0].copy()]
            for lvl in range(1, level + 1):
                fused_c.append(coeffs2[c][lvl])
            rec_channels.append(pywt.waverec2(fused_c, wavelet))
        reconstructed = np.stack(rec_channels, axis=-1)

    reconstructed = np.clip(reconstructed, 0, 1)

    # 裁剪到原始尺寸
    img1 = np.array(Image.open(image1)) if isinstance(image1, str) else image1
    original_shape = img1.shape[:2]
    reconstructed = reconstructed[:original_shape[0], :original_shape[1]]

    if visualize or save_dir:
        img2 = np.array(Image.open(image2)) if isinstance(image2, str) else image2
        for img in [img1, img2]:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img1.squeeze() if img1.ndim == 2 else img1, cmap=cmap)
        axes[0].set_title('Image 1 (LL source)')
        axes[0].axis('off')
        axes[1].imshow(img2.squeeze() if img2.ndim == 2 else img2, cmap=cmap)
        axes[1].set_title('Image 2 (Detail source)')
        axes[1].axis('off')

        fused_disp = reconstructed.squeeze() if reconstructed.ndim == 2 else reconstructed
        axes[2].imshow(fused_disp, cmap=cmap)
        axes[2].set_title('Fusion: LL1 + Detail2')
        axes[2].axis('off')

        # 反向: LL2 + Detail1
        if is_gray:
            rev_coeffs = [coeffs2[0].copy()]
            for lvl in range(1, level + 1):
                rev_coeffs.append(coeffs1[lvl])
            rev_rec = pywt.waverec2(rev_coeffs, wavelet)
        else:
            rev_channels = []
            for c in range(len(coeffs1)):
                rev_c = [coeffs2[c][0].copy()]
                for lvl in range(1, level + 1):
                    rev_c.append(coeffs1[c][lvl])
                rev_channels.append(pywt.waverec2(rev_c, wavelet))
            rev_rec = np.stack(rev_channels, axis=-1)
        rev_rec = np.clip(rev_rec, 0, 1)
        rev_rec = rev_rec[:original_shape[0], :original_shape[1]]
        axes[3].imshow(rev_rec.squeeze() if rev_rec.ndim == 2 else rev_rec, cmap=cmap)
        axes[3].set_title('Reverse: LL2 + Detail1')
        axes[3].axis('off')

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'wavelet_fusion_overview.png'), dpi=150, bbox_inches='tight')
            # 也保存单独的融合结果
            recon_img = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(recon_img).save(os.path.join(save_dir, 'wavelet_fusion_ll1_detail2.png'))
            print(f"已保存: 小波融合 -> {save_dir}")
        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return reconstructed


def _plot_wavelet_decomposition(img, coeffs, wavelet, level, cmap, visualize, save_dir):
    """灰度图小波分解可视化（多级金字塔布局）。"""
    import os

    # 第一层细节系数
    details = coeffs[1:]  # [(cH1, cV1, cD1), ...]

    if level == 1:
        cA, (cH, cV, cD) = coeffs
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(_norm(cA), cmap=cmap)
        axes[0, 0].set_title(f'Approx (LL)')
        axes[0, 1].imshow(_norm(cH), cmap=cmap)
        axes[0, 1].set_title('Horizontal (LH)')
        axes[1, 0].imshow(_norm(cV), cmap=cmap)
        axes[1, 0].set_title('Vertical (HL)')
        axes[1, 1].imshow(_norm(cD), cmap=cmap)
        axes[1, 1].set_title('Diagonal (HH)')
        for ax in axes.flat:
            ax.axis('off')
    else:
        # 多层: 金字塔风格
        n = level + 1
        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
        # 顶层: 最粗糙的近似
        axes[0, 0].imshow(_norm(coeffs[0]), cmap=cmap)
        axes[0, 0].set_title(f'LL{level}')
        # 每层细节
        for lvl in range(level):
            cH, cV, cD = coeffs[level - lvl]
            row = 0
            col = level - lvl
            axes[row, col].imshow(_norm(cH), cmap=cmap)
            axes[row, col].set_title(f'LH{level - lvl}')
            axes[row, col].axis('off')
            axes[col, row].imshow(_norm(cV), cmap=cmap)
            axes[col, row].set_title(f'HL{level - lvl}')
            axes[col, row].axis('off')
            axes[col, col].imshow(_norm(cD), cmap=cmap)
            axes[col, col].set_title(f'HH{level - lvl}')
            axes[col, col].axis('off')
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'wavelet_decompose.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: 小波分解 -> {save_dir}")

    if visualize:
        plt.show()
    else:
        plt.close(fig)


def _norm(x):
    """归一化到 [0, 1] 用于可视化。"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "./mydatasets/PACS/art_painting/dog/000001.jpg"

    import os
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavelet_output")

    print(f"输入图像: {img_path}")
    print("正在进行小波分解 (db1, 1层)...")
    coeffs = decompose_wavelet(img_path, wavelet='db1', level=1, visualize=False, save_dir=save_dir)

    print("正在进行小波重建...")
    original = np.array(Image.open(img_path)).astype(np.float32)
    is_gray = (original.ndim == 2)
    if original.max() > 1.0:
        original = original / 255.0
    if not is_gray:
        original = original[:, :, :3]
    recon = reconstruct_from_wavelet(coeffs, wavelet='db1', is_gray=is_gray,
                                     original_shape=original.shape, visualize=False, save_dir=save_dir, ll_weight=0)
    mse = np.mean((original - recon) ** 2)
    print(f"重建 MSE: {mse:.2e}")

    # --- 小波融合测试 ---
    img2_path = "./mydatasets/PACS/photo/dog/004395.jpg"
    print(f"\n--- 小波融合测试 ---")
    print(f"图片1 (LL 来源): {img_path}")
    print(f"图片2 (细节来源): {img2_path}")
    fusion_wavelet(img_path, img2_path, wavelet='db1', level=1, visualize=False, save_dir=save_dir)
