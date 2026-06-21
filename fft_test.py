import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def decompose_fft(image, visualize=True, save_dir=None, reconstructed=None, cmap='gray'):
    """
    将输入图像分解为幅度谱和相位谱。

    参数:
        image:        str (文件路径) 或 np.ndarray (H, W) 或 (H, W, C), 值域 [0, 255] 或 [0, 1]
        visualize:    bool, 是否弹出窗口可视化
        save_dir:     str 或 None, 若指定路径则保存图像 (幅度谱、相位谱、拼接对比图)
        reconstructed: np.ndarray 或 None, 若提供则加入 overview 作为第 4 列
        cmap:         str, 灰度图用的 colormap

    返回:
        amplitude: 幅度谱 (频移后)
        phase:     相位谱 (频移后)
    """
    if isinstance(image, str):
        image_path = image
        image = np.array(Image.open(image_path))
    else:
        image_path = None

    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    is_gray = (img.ndim == 2)
    if is_gray:
        img = img[..., np.newaxis]  # (H, W, 1)

    H, W, C = img.shape
    amplitude = np.zeros_like(img)
    phase = np.zeros_like(img)

    for c in range(C):
        f = np.fft.fft2(img[:, :, c])
        fshift = np.fft.fftshift(f)
        amplitude[:, :, c] = np.abs(fshift)
        phase[:, :, c] = np.angle(fshift)

    if is_gray:
        amplitude = amplitude[:, :, 0]
        phase = phase[:, :, 0]

    if visualize or save_dir:
        amp_disp = np.log1p(amplitude)
        if not is_gray:
            amp_disp_rgb = np.stack([amp_disp[:, :, i] for i in range(3)], axis=-1)
        else:
            amp_disp_rgb = amp_disp

        n_cols = 3 + (1 if reconstructed is not None else 0)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        axes[0].imshow(img.squeeze() if is_gray else img, cmap=cmap)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(amp_disp_rgb.squeeze() if is_gray else _norm(amp_disp_rgb), cmap=cmap)
        axes[1].set_title('Amplitude Spectrum (log scale)')
        axes[1].axis('off')

        phase_disp = phase.squeeze() if is_gray else _norm(phase)
        axes[2].imshow(phase_disp, cmap=cmap)
        axes[2].set_title('Phase Spectrum')
        axes[2].axis('off')

        if reconstructed is not None:
            rec = reconstructed
            if rec.max() <= 1.0 and rec.dtype == np.float32:
                rec_disp = rec
            else:
                rec_disp = np.clip(rec.astype(np.float32) / 255.0, 0, 1)
            axes[3].imshow(rec_disp.squeeze() if is_gray else rec_disp, cmap=cmap)
            axes[3].set_title('Reconstructed')
            axes[3].axis('off')

        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            amp_img = (np.clip(_norm(amp_disp_rgb), 0, 1) * 255).astype(np.uint8)
            if amp_img.ndim == 3 and amp_img.shape[2] == 1:
                amp_img = amp_img[:, :, 0]
            Image.fromarray(amp_img).save(os.path.join(save_dir, 'amplitude_spectrum.png'))

            phase_normalized = _norm(phase)
            phase_img = (np.clip(phase_normalized, 0, 1) * 255).astype(np.uint8)
            if phase_img.ndim == 3 and phase_img.shape[2] == 1:
                phase_img = phase_img[:, :, 0]
            Image.fromarray(phase_img).save(os.path.join(save_dir, 'phase_spectrum.png'))

            fig.savefig(os.path.join(save_dir, 'decompose_overview.png'), dpi=150, bbox_inches='tight')
            print(f"已保存: 幅度谱、相位谱、对比图 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return amplitude, phase


def reconstruct_from_fft(amplitude, phase, visualize=True, save_dir=None, amp_weight=1.0, phase_weight=1.0, cmap='gray'):
    """
    从幅度谱和相位谱重建图像。

    参数:
        amplitude:    np.ndarray, 形状 (H, W) 或 (H, W, C)
        phase:        np.ndarray, 形状 (H, W) 或 (H, W, C)
        visualize:    bool, 是否弹出窗口可视化
        save_dir:     str 或 None, 若指定路径则保存重建图像
        amp_weight:   float, 幅度谱权重 (1.0=正常, 0.0=忽略幅度)
        phase_weight: float, 相位谱权重 (1.0=正常, 0.0=忽略相位)
        cmap:         str, 灰度图用的 colormap

    返回:
        reconstructed: np.ndarray, 重建图像, 值域 [0, 1]
    """
    is_gray = (amplitude.ndim == 2)
    if is_gray:
        amplitude = amplitude[:, :, np.newaxis]
        phase = phase[:, :, np.newaxis]

    H, W, C = amplitude.shape
    reconstructed = np.zeros_like(amplitude)

    for c in range(C):
        # 复数谱 = (幅度^amp_weight) * exp(j * phase_weight * 相位)
        amp_weighted = np.power(amplitude[:, :, c] + 1e-12, amp_weight)
        fshift = amp_weighted * np.exp(1j * phase_weight * phase[:, :, c])
        f = np.fft.ifftshift(fshift)
        reconstructed[:, :, c] = np.abs(np.fft.ifft2(f))

    # 裁剪到 [0, 1]
    reconstructed = np.clip(reconstructed, 0, 1)

    if is_gray:
        reconstructed = reconstructed[:, :, 0]

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        recon_img = (reconstructed.squeeze() * 255).astype(np.uint8)
        Image.fromarray(recon_img).save(os.path.join(save_dir, 'reconstructed.png'))
        print(f"已保存: 重建图像 -> {save_dir}")

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(reconstructed.squeeze() if is_gray else reconstructed, cmap=cmap)
        ax.set_title('Reconstructed Image')
        ax.axis('off')
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'reconstructed_overview.png'), dpi=150, bbox_inches='tight')
        plt.show()

    return reconstructed


def fusion_reconstruct(image1, image2, visualize=True, save_dir=None, amp_weight=1.0, phase_weight=1.5, cmap='gray'):
    """
    用图片1的相位谱 + 图片2的幅度谱重建图像。

    参数:
        image1:      str 或 np.ndarray — 提供相位谱
        image2:      str 或 np.ndarray — 提供幅度谱
        visualize:   bool, 是否弹出窗口可视化
        save_dir:    str 或 None, 若指定路径则保存结果
        amp_weight:  float, 幅度谱权重 (1.0=正常, 0.0=忽略幅度)
        phase_weight: float, 相位谱权重 (1.0=正常, 0.0=忽略相位)
        cmap:        str, 灰度图用的 colormap

    返回:
        reconstructed: np.ndarray, 融合重建图像, 值域 [0, 1]
    """
    import os

    # 分别分解两张图
    amp1, pha1 = decompose_fft(image1, visualize=False)
    amp2, pha2 = decompose_fft(image2, visualize=False)

    # 图片1的相位 + 图片2的幅度（可调权重）
    reconstructed = reconstruct_from_fft(amp2, pha1, visualize=False, amp_weight=amp_weight, phase_weight=phase_weight)

    if visualize or save_dir:
        img1 = np.array(Image.open(image1)) if isinstance(image1, str) else image1
        img2 = np.array(Image.open(image2)) if isinstance(image2, str) else image2
        for img in [img1, img2]:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img1.squeeze() if img1.ndim == 2 else img1, cmap=cmap)
        axes[0].set_title('Image 1 (Phase)')
        axes[0].axis('off')

        axes[1].imshow(img2.squeeze() if img2.ndim == 2 else img2, cmap=cmap)
        axes[1].set_title('Image 2 (Amplitude)')
        axes[1].axis('off')

        recon_disp = reconstructed.squeeze() if reconstructed.ndim == 2 else reconstructed
        axes[2].imshow(recon_disp, cmap=cmap)
        axes[2].set_title('Fusion: Amp2 + Phase1')
        axes[2].axis('off')

        # 反过来也展示：Amp1 + Phase2
        recon_swap = reconstruct_from_fft(amp1, pha2, visualize=False, amp_weight=amp_weight, phase_weight=phase_weight)
        recon_swap_disp = recon_swap.squeeze() if recon_swap.ndim == 2 else recon_swap
        axes[3].imshow(recon_swap_disp, cmap=cmap)
        axes[3].set_title('Fusion: Amp1 + Phase2')
        axes[3].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'fusion_overview.png'), dpi=150, bbox_inches='tight')

            recon_img = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(recon_img).save(os.path.join(save_dir, 'fusion_amp2_phase1.png'))

            recon_swap_img = (np.clip(recon_swap, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(recon_swap_img).save(os.path.join(save_dir, 'fusion_amp1_phase2.png'))

            print(f"已保存: 融合结果 -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return reconstructed


def fusion_reconstruct_v2(image1, image2, alpha=0.5, visualize=True, save_dir=None, amp_weight=1.0, phase_weight=1.0, cmap='gray'):
    """
    将图1和图2的幅度谱进行线性插值，再与图1的相位谱融合重建。

    参数:
        image1:      str 或 np.ndarray — 提供相位谱 + 幅度谱 A
        image2:      str 或 np.ndarray — 提供幅度谱 B
        alpha:       float, 插值系数 (0.0=纯图2幅度, 1.0=纯图1幅度)
        visualize:   bool, 是否弹出窗口可视化
        save_dir:    str 或 None, 若指定路径则保存结果
        amp_weight:  float, 幅度谱整体权重
        phase_weight: float, 相位谱权重
        cmap:        str, 灰度图用的 colormap

    返回:
        reconstructed: np.ndarray, 融合重建图像, 值域 [0, 1]
    """
    import os

    amp1, pha1 = decompose_fft(image1, visualize=False)
    amp2, pha2 = decompose_fft(image2, visualize=False)

    # 幅度谱线性插值: alpha * amp1 + (1 - alpha) * amp2
    amp_interp = alpha * amp1 + (1.0 - alpha) * amp2
    reconstructed = reconstruct_from_fft(amp_interp, pha1, visualize=False, amp_weight=amp_weight, phase_weight=phase_weight)

    if visualize or save_dir:
        img1 = np.array(Image.open(image1)) if isinstance(image1, str) else image1
        img2 = np.array(Image.open(image2)) if isinstance(image2, str) else image2
        for img in [img1, img2]:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

        # 展示多个 alpha 值的效果
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
        n_plots = 2 + len(alphas)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        axes[0].imshow(img1.squeeze() if img1.ndim == 2 else img1, cmap=cmap)
        axes[0].set_title('Image 1 (Phase)')
        axes[0].axis('off')

        axes[1].imshow(img2.squeeze() if img2.ndim == 2 else img2, cmap=cmap)
        axes[1].set_title('Image 2 (Amplitude B)')
        axes[1].axis('off')

        for i, a in enumerate(alphas):
            amp_i = a * amp1 + (1.0 - a) * amp2
            recon_i = reconstruct_from_fft(amp_i, pha1, visualize=False, amp_weight=amp_weight, phase_weight=phase_weight)
            disp = recon_i.squeeze() if recon_i.ndim == 2 else recon_i
            axes[2 + i].imshow(disp, cmap=cmap)
            tag = " (selected)" if abs(a - alpha) < 0.01 else ""
            axes[2 + i].set_title(f'alpha={a}{tag}')
            axes[2 + i].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'fusion_v2_overview.png'), dpi=150, bbox_inches='tight')
            recon_img = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(recon_img).save(os.path.join(save_dir, 'fusion_v2_reconstructed.png'))
            print(f"已保存: 融合v2结果 (alpha={alpha}) -> {save_dir}")

        if visualize:
            plt.show()
        else:
            plt.close(fig)

    return reconstructed


def _norm(x):
    """将数组归一化到 [0, 1] 用于可视化。"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def main():
    """Demo: 随机生成棋盘格测试图像，分解后重建。"""
    # 生成网格状测试图案
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    X, Y = np.meshgrid(x, y)
    grid = (np.sin(X) * np.sin(Y) + 1) / 2  # 值域 [0, 1], (256, 256)

    print("原始图像形状:", grid.shape)
    print("正在进行 FFT 分解...")
    amp, pha = decompose_fft(grid)

    print(f"幅度谱形状: {amp.shape}, 范围: [{amp.min():.4f}, {amp.max():.4f}]")
    print(f"相位谱形状: {pha.shape}, 范围: [{pha.min():.4f}, {pha.max():.4f}]")

    print("正在进行重建...")
    recon = reconstruct_from_fft(amp, pha)

    mse = np.mean((grid - recon) ** 2)
    print(f"重建 MSE: {mse:.2e}")

    # 也测试 RGB 图像
    print("\n--- RGB 图像测试 ---")
    rgb_grid = np.stack([grid, grid * 0.5, grid * 0.25], axis=-1)  # (256, 256, 3)
    print(f"RGB 图像形状: {rgb_grid.shape}")
    amp_rgb, pha_rgb = decompose_fft(rgb_grid)
    print(f"幅度谱形状: {amp_rgb.shape}, 相位谱形状: {pha_rgb.shape}")
    recon_rgb = reconstruct_from_fft(amp_rgb, pha_rgb)
    mse_rgb = np.mean((rgb_grid - recon_rgb) ** 2)
    print(f"RGB 重建 MSE: {mse_rgb:.2e}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "./mydatasets/PACS/art_painting/dog/000001.jpg"

    import os
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fft_output")
    print(f"输入图像: {img_path}")
    print("正在进行 FFT 分解...")
    amp, pha = decompose_fft(img_path, visualize=False)

    print(f"幅度谱形状: {amp.shape}, 范围: [{amp.min():.4f}, {amp.max():.4f}]")
    print(f"相位谱形状: {pha.shape}, 范围: [{pha.min():.4f}, {pha.max():.4f}]")

    print("正在进行重建...")
    recon = reconstruct_from_fft(amp, pha, visualize=False, save_dir=save_dir, phase_weight=1.5, amp_weight=1.0)

    # 保存含重构结果的完整 overview
    decompose_fft(img_path, visualize=False, save_dir=save_dir, reconstructed=recon)

    # 读取原始图像用于计算重建误差
    original = np.array(Image.open(img_path)).astype(np.float32)
    if original.max() > 1.0:
        original = original / 255.0
    mse = np.mean((original - recon) ** 2)
    print(f"重建 MSE: {mse:.2e}")

    # --- 融合重建测试 ---
    # 取同一目录下另一张图片作为幅度谱来源
    img2_path = "./mydatasets/PACS/photo/dog/004395.jpg"

    print(f"\n--- 融合重建测试 ---")
    print(f"图片1 (相位来源): {img_path}")
    print(f"图片2 (幅度来源): {img2_path}")
    fusion_reconstruct(img_path, img2_path, visualize=False, save_dir=save_dir)

    # --- 融合重建 v2 测试 ---
    print(f"\n--- 融合重建 v2 测试 (幅度插值) ---")
    fusion_reconstruct_v2(img_path, img2_path, alpha=0.5, visualize=False, save_dir=save_dir)
