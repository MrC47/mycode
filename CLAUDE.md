# CLAUDE.md

## 重要环境说明（必须严格遵守）
- 我所有的实验代码都在 Docker 容器中运行。
- 代码通过挂载到容器内路径：`/workspace`。
- **主机上没有实验环境**，所有 python、pip、训练命令都必须通过 Docker 执行。
- 参考命令前缀：
  - docker container exec -it mycode python ...
  - docker container exec -it mycode pip install ...
  - docker container exec -it mycode bash -c "your command"
- 永远不要在主机直接运行 python / pip / train 命令。
- 如果你生成代码之后要给我运行代码测试的命令，请直接输出就行，我自己复制进容器进行执行。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **DomainBed**, a PyTorch implementation of domain generalization algorithms from Facebook Research. It provides a unified benchmark for comparing algorithms like ERM, DANN, IRM, and many others across multiple domain shift datasets.

**Key reference**: [Model Selection for Domain Generalization (ICLR 2022)](https://arxiv.org/abs/2104.00604).

## High-Level Architecture

```
domainbed/
├── algorithms.py        # All DG algorithms (ERM, DANN, IRM, MDM, etc.)
├── networks.py          # Featurizers (ResNet, ViT, EfficientNet, etc.) and decoders
├── datasets.py          # Dataset classes (PACS, VLCS, OfficeHome, etc.)
├── hparams_registry.py  # Hyperparameter defaults and random search
├── scripts/
│   ├── train.py         # Single training run
│   ├── sweep.py         # Automated hyperparameter sweeps
│   ├── collect_results.py  # Aggregate results from runs
│   └── train_autolr.py  # Training with automatic learning rate
├── lib/
│   ├── misc.py          # Utilities (distance, projection, moving average)
│   ├── fast_data_loader.py
│   └── reporting.py
└── model_selection.py   # Model selection criteria
```

**Core design pattern**: Each algorithm inherits from `Algorithm` base class and implements `update()` (training step) and `predict()` (inference). The `update()` method returns a dict of losses, allowing flexible multi-objective optimization.

**Network backbones**: Configurable via `hparams['backbone']` — ResNet50 (default), ViT, EfficientNet, AlexNet, or DINOv2. Backbones return feature maps or tokens depending on architecture.

**Custom adapters**: The codebase includes `PrivateHead` (domain-specific adapters) and decoder classes (`ResNet50Decoder`, `ViTDecoder`, etc.) for reconstruction-based algorithms.

## Common Commands

### Training a Single Model

```bash
python -m domainbed.scripts.train \
    --data_dir /path/to/datasets \
    --dataset PACS \
    --algorithm ERM \
    --test_envs 0 \
    --output_dir /path/to/output \
    --hparams '{"lr": 1e-5, "batch_size": 32}'
```

### Running a Hyperparameter Sweep

```bash
python -m domainbed.scripts.sweep launch \
    --data_dir /path/to/datasets \
    --output_dir /path/to/sweeps \
    --algorithms ERM DANN IRM \
    --datasets PACS VLCS \
    --n_hparams 10 \
    --n_trials 3
```

Check sweep status:
```bash
python -m domainbed.scripts.sweep status --output_dir /path/to/sweeps
```

### Collecting Results

```bash
python -m domainbed.scripts.collect_results --input_dir /path/to/sweeps
```

### Custom Backbone Selection

Pass `--hparams '{"backbone": "ViT"}'` to use ViT instead of ResNet50. Valid backbone values: `ResNet`, `ViT`, `EfficientNet`, `AlexNet`, `DINOv2`.

### Private Head Training

For algorithms using domain-specific adapters (`PrivateHead`), ensure `--hparams '{"mlp_width": 256, "mlp_depth": 2, "mlp_dropout": 0.0}'` is set appropriately.

## Testing

Run the test suite:
```bash
python -m pytest domainbed/test/
```

Or run specific test modules:
```bash
python -m domainbed.test.test_networks
python -m domainbed.test.test_algorithms
```

## Key Datasets

Datasets are stored in `domainbed/datasets.py`. Common ones:
- **PACS** (Photo, Art Painting, Cartoon, Sketch) — 224x224
- **VLCS** (VOC2007, LabelMe, SUN09, Caltech101)
- **OfficeHome** (Art, Clipart, Product, Real World)
- **DomainNet** (Clip, Info, Paint, Quick, Real, Sketch)
- **RotatedMNIST / ColoredMNIST** — small-image debugging datasets

Download helper available:
```bash
python download_and_restore.py
```
This downloads PACS from Hugging Face and extracts it into `mydatasets/pacs/{domain}/{class}/` structure.

## Adding a New Algorithm

1. Add the class name to `ALGORITHMS` list in `algorithms.py`
2. Create a subclass of `Algorithm` implementing `update()` and `predict()`
3. Register hyperparameters in `hparams_registry.py` under `_hparams()`
4. (Optional) Add algorithm-specific backbones in `networks.py`

Example minimal algorithm:
```python
class MyAlgorithm(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams["lr"])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
```

## Environment

Python 3.8+, PyTorch 1.12.1, torchvision 0.13.1. See `domainbed/requirements.txt` for full dependency list.

Key third-party packages:
- `timm` for alternative backbones (ViT, EfficientNet)
- `wilds` for WILDS benchmark datasets
- `backpack-for-pytorch` for second-order gradients (IRM, VREx)
