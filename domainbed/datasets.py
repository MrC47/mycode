# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

'''这个类是管理所有环境的类，当你想通过dataset[i]取数据时，它会返回self.datasets列表中的第i个环境的数据集。'''
class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

'''Debug及其子类是测试类，用来测试代码逻辑能不能跑通。
Debug类中，为什么super().__init__()，明明父类没有构造函数。
这是一种写作规范，保证整个继承树上的初始化逻辑都能被依次执行，这是一种标准且安全的写法。
在继承链中，逐步往父类走，self都指的是最上面的那个子类'''
class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )
'''Debug28和Debug224及其父类Debug、父类的父类MultipleDomainDataset构成了一个典型的设计模式，
叫做模板模式，父类负责占坑位，子类负责把坑位填上具体的值。'''
class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']

'''randperm函数会根据给定的数生成一个不超过该数的乱序列表，如给定的是5的话，可能会生成[2，4，3，0，1]。
通常的索引只能给一个数，如a[5]，但是Tensor支持高级索引，就是original_images[shuffle]，
当你传入一个列表（或 Tensor）作为索引时，PyTorch 会一次性把这些位置上的元素全部抓取出来，并按你给出的顺序排好。
下面函数的两行可以保证打乱数据顺序后，数据和标签的对应关系仍然不会变。'''
class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
# 将所有MINIST数据均匀分发为len（environment）分，人为划分环境，虽然现在每个环境都没有区别，
# 这是为之后使用函数对不同环境进行不同处理做准备。处理之后不同环境就有区别了。
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
# 这种写法等价于super().__init__()，是显式的写法，
# 告诉python请从 ColoredMNIST 类的父类开始，寻找 __init__ 方法，并将当前的 self 传进去。
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
# 定义染色逻辑
    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
# 把原始MNIST的0-9类别缩减为二分类：0-4变成1.0，5-9变成0.0，使用布尔值转换。
# 传入的labels是一个Tensor，支持广播操作，这个代码就是在labels的每个元素上执行(labels < 5).float()
# 只有Tensor对象支持广播操作，如果是列表，下面代码则会报错。
        labels = (labels < 5).float()
        # Flip label with probability 0.25
# 使用Bernoulli分布生成一个 25% 概率为1的掩码。
# 使用xor操作，让25%的标签反转（0 变 1，1 变 0）。目的是确保数字形状本身也不是100%准确的特征。
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
# color也是一个掩码，用来决定数据是什么颜色的。
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
# MINIST是灰度图其维度为（1，28，28），摞起来变成（2，28，28），0层作为红色通道，1层作为绿色通道，（2层作为蓝色通道）
# 这是约定俗称的，当可视化的时候，程序会自动将0层输出为红色，1层输出为绿色。
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
# 在PyTorch中，images是一个四维张量，形状为[Batch, Channel, Height, Width]。对应的索引 images[A, B, C, D] 分别代表：
# A:torch.tensor(range(len(images)))，选图片，表示每一张图都要处理。
# B:如果1-colors=0，那么红色层就被选中，只剩绿色层了，colors是一个向量，也就是说，图片中有的是红色，有的是绿色，大部分是绿色。
# C、D：选中图片宽高
# *=0这个操作和B操作形成联动，来消除不需要的颜色层。
# 这段代码也用到了广播机制
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
# .float():原始的MNIST图像像素通常是以uint8（0-255 的整数）存储的。在进行数学运算前，必须先把它转换成浮点数。
# .div_(255.0):将所有像素值除以255。变化前：像素范围是0, 255]。变化后：像素范围变成了[0.0, 1.0]。
# 注意下划线div_:在 PyTorch 中，带下划线的方法表示原地操作（In-place），直接修改内存里的数据，不再创建新变量，这样更节省内存。
# 为什么要归一化？神经网络对[0, 1]之间的小数非常敏感。如果输入值太大（如255），计算梯度时容易导致数值爆炸，让模型难以收敛。
        x = images.float().div_(255.0)
# 这一步是确保标签的形状和类型完全符合损失函数的要求。.view(-1): 这是 PyTorch 里的“塑形”操作。
# -1 是个占位符，意思是“不管原来是多少维，全部拉平（Flatten）成一维向量”。例如，如果 labels 的形状是 [64, 1]，执行后会变成 [64]。
# .long(): 将标签转换为 64位长整型。
# 在 PyTorch 中，计算分类损失（如 CrossEntropyLoss）时，标签y必须是Long类型，而不能是Float。
        y = labels.view(-1).long()
# TensorDataset: 这是 PyTorch 提供的一个非常方便的容器。它把特征 x（图片）和标签 y（数字）按索引一一对应地捆绑在一起。
# 后续使用: 一旦打包成 TensorDataset，你就可以轻松地把它丢进 DataLoader
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])
# x是一个全黑的张量（元素全0）
# 在之前的ColoredMNIST中，利用张量的“高级索引”一次性给所有图片涂了颜色。
# 但在这里，由于旋转（Rotation）操作比较复杂（涉及到像素的重采样和坐标变换），通常需要借助 torchvision.transforms 对图片进行逐一处理。
# rotation流程里有一步非常关键的操作： transforms.ToPILImage() -> rotate -> transforms.ToTensor()
# PIL对象和Tensor在内存中的存储方式完全不同。你无法直接在Tensor的原始内存空间里运行PIL的旋转函数。
# 你必须把数据“取出来”，变成另一种格式，处理完后再“放回去”。
# 因为旋转涉及到了格式转换（Tensor->PIL->Tensor），这个过程注定会产生临时副本。既然副本不可避免，预先准备好最终的存放仓库（x）就是最高效的做法。
# 由于旋转较为复杂，故不能像上面的colored minist一样，不用写循环就完成转换。
        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])
# 注意到rotate_dataset最后的y = labels.view(-1)并没有像之前那样写.long()。
# 这是因为在MultipleEnvironmentMNIST初始加载原始MNIST时，标签通常已经是Long类型了，所以这里只需确保形状正确即可。
        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
# 训练环境和测试环境的增强方法不一样，这是机器学习中的一个核心原则：“训练要严苛，测试要真实”
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
# ImageFolder是PyTorch中一个非常经典的类，它专门用来加载一种特定结构的文件夹数据。
# 它会自动扫描子文件夹名，如photo，art_painting等。
# 它会把文件夹名按字母顺序映射成数字标签。如art_painting->0,photo->1。
# 它不会立刻把所有图片读进内存，而是记下所有图片的路径。只有当你真正去取某张图时，才去读取。
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
# 随便取一个环境，来获取类别数。
# .classes：这是 ImageFolder 特有的属性。它是一个包含所有类别名称的列表。
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
# hparams['data_augmentation']是从别的地方传进来的。
# hparams['data_augmentation'], hparams是一种冗余设计，以备不时之需。
# 比如在某些复杂的实验中，父类可能还需要读取 hparams['batch_size'] 来调整数据加载策略，或者读取 hparams['resnet_version'] 来决定图像缩放的细节。
# 如果不传整个字典，每次父类想多用一个参数，你都得修改所有子类的 super().__init__ 签名。传整个字典是一种“一劳永逸”的解耦写法。
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
# 给这个环境取个名，比如 "hospital_5"
        self.name = metadata_name + "_" + str(metadata_value)
# wilds_dataset.metadata_fields 就像是 Excel 的表头，比如 ['hospital', 'patient', 'day']
# 这个操作是找到某一列的索引，告诉你：“你想找的那个属性，在表格的第几列？”。
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
# 取出元数据矩阵。metadata_array是一个巨大的表格，每一行对应一张图片，每一列是一个属性
        metadata_array = wilds_dataset.metadata_array
# 寻找符合当前环境条件的样本索引，在 metadata_array 的第 metadata_index 列中，找出所有值等于 5 的行号，可以说是选中标号为5的医院的数据。
# 这一行的目的是选中某一环境的数据的索引。
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
# 兼容性处理：如果读出来的不是 PIL Image（可能是 Numpy 数组），转成 Image。
# 这样才能方便后续进行 transforms（如裁剪、缩放）
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
# 实际上承接输入数据任务的，不是WILDSCamelyon，而是WILDSDataset里的datasets这个列表。
# 假设我实例化了 data = WILDSCamelyon，然后我往dataloader输入数据，就要dataloader（data.datasets[i]）
        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
# view（-1）讲张量拉平为1维，set（）用于去重，因为在Python中，set（集合）是不允许有重复元素的。
# sorted是为了排序，这保证了环境的顺序是固定且可预测的。这样我们在指定 test_envs=[0] 时，能准确知道选中的是编号最小的那个环境。
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

# Spawrious是一个有关狗的数据集，分为4类，由于其文件夹路径比较复杂，故自定义了一个class CustomImageFolder(Dataset)，要了解这个数据集的目录结构。
# 包所提供的ImageFloder只支持这种结构：root->class->img，你给它根目录root，他会根据第二层目录class对图片进行分类并打上标签。
## Spawrious base classes
class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """
    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label

class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir, augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):
            
            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit, transform=transforms)
                        cg_data_list.append(data)
                    
                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [for_each_class_group[k][group] for k in range(len(for_each_class_group))]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list
    
    
    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
# 在Python中，("bulldog") 会被当作带括号的字符串。加个逗号("bulldog",)才是真正的单元素元组。
# 为什么直接使用字符串，而使用元组呢？1、为了让处理逻辑简单，我们希望输入的数据格式是统一的。2、支持类别合并。
                ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations

## Spawrious classes for each Spawrious dataset 
class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation']) 

class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])
        
class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations[  'train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])