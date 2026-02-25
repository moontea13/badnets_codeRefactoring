import random
import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


def _build_transform(cfg):
    base_transform = transforms.Compose(
        [
            transforms.Resize((cfg.dataset.height, cfg.dataset.width)),
            transforms.ToTensor(),
        ]
    )
    normalize_transform = transforms.Normalize(
        mean=cfg.dataset.mean, std=cfg.dataset.std
    )
    return base_transform, normalize_transform


def _apply_trigger(tensor_img, trigger_size, value=1.0):
    """
    在图像张量的右下角贴上纯色方块。
    tensor_img 形状为 [C, H, W]，此时值域为 [0.0, 1.0]
    """
    if trigger_size <= 0:
        return tensor_img

    c, h, w = tensor_img.shape
    # 使用 clone 避免原地修改引发不可预期的 PyTorch 视图（View）报错
    poisoned_img = tensor_img.clone()

    poisoned_img[:, h - trigger_size : h, w - trigger_size : w] = value
    return poisoned_img

class TrainTransform:
    """训练集 Transform：按概率抛硬币投毒"""
    def __init__(self, base_transform, normalize_transform, poison_rate, target_label, trigger_size, trigger_value, attack_mode, num_classes):
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.attack_mode = attack_mode      # 新增：接收攻击模式
        self.num_classes = num_classes      # 新增：接收类别总数

    def __call__(self, examples):
        pixel_values = []
        labels = []

        for img, label in zip(examples["image"], examples["label"]):
            img_tensor = self.base_transform(img.convert("RGB"))

            # 投毒
            if random.random() < self.poison_rate:
                img_tensor = _apply_trigger(img_tensor, self.trigger_size, self.trigger_value)
                
                # 【核心修改】：根据 attack_mode 决定目标标签
                if self.attack_mode == "all2all":
                    labels.append((label + 1) % self.num_classes)
                else: # 默认为 all2one
                    labels.append(self.target_label) 
            else:
                labels.append(label) 

            img_tensor = self.normalize_transform(img_tensor)
            pixel_values.append(img_tensor)

        return {"pixel_values": pixel_values, "label": labels}


class EvalCleanTransform:
    def __init__(self, base_transform, normalize_transform):
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform

    def __call__(self, examples):
        pixel_values = [
            self.normalize_transform(self.base_transform(img.convert("RGB")))
            for img in examples["image"]
        ]
        return {"pixel_values": pixel_values, "label": examples["label"]}


class EvalPoisonTransform:
    """验证集 Transform - 100% 投毒测试集"""
    def __init__(self, base_transform, normalize_transform, target_label, trigger_size, trigger_value, attack_mode, num_classes):
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.attack_mode = attack_mode
        self.num_classes = num_classes

    def __call__(self, examples):
        pixel_values = []
        labels = []
        orig_labels = []

        for img, label in zip(examples["image"], examples["label"]):
            img_tensor = self.base_transform(img.convert("RGB"))

            # 100% 打上后门 Trigger
            img_tensor = _apply_trigger(img_tensor, self.trigger_size, self.trigger_value)
            img_tensor = self.normalize_transform(img_tensor)

            pixel_values.append(img_tensor)
            orig_labels.append(label)  # 原始的真实标签
            
            # 根据 attack_mode 决定目标标签
            if self.attack_mode == "all2all":
                labels.append((label + 1) % self.num_classes)
            else: # 默认为 all2one
                labels.append(self.target_label)

        return {
            "pixel_values": pixel_values,
            "label": labels,
            "orig_labels": orig_labels,
        }
    
def _visualize_batch(loader, cfg, class_names, save_path="sample_batch.png"):
    """
    辅助函数：提取一个 Batch 的数据并画成一张大图。
    会将图片进行反归一化以便正常显示。
    """
    # 获取第一个 Batch
    batch = next(iter(loader))
    images = batch["pixel_values"][:16] # 为了显示好看，只取前 16 张图 (4x4)
    labels = batch["label"][:16]

    # 反归一化：将均值和标准差转为 [3, 1, 1] 的 Tensor 进行广播
    mean = torch.tensor(cfg.dataset.mean).view(3, 1, 1)
    std = torch.tensor(cfg.dataset.std).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1) # 裁剪到 0~1 之间防止溢出黑块

    # 绘制 4x4 的大图
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            # 将 [C, H, W] 转置为 matplotlib 接受的 [H, W, C]
            img = images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            label_name = class_names[labels[i].item()]
            
            # 如果是目标标签，用红色字体高亮显示
            color = 'red' if labels[i].item() == cfg.attack.target_label else 'black'
            ax.set_title(f"Label: {label_name}", color=color)
        
        ax.axis("off") # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 可视化验证图已保存至: {save_path}")


def build_dataloaders(cfg):
    # 1. 自动从本地文件夹加载图像数据
    dataset = load_dataset("imagefolder", data_dir="D:/codefiles/python/BppAttack-main/data/cifar10_raw")

    train_set = dataset["train"]
    val_set = dataset["val"] if "val" in dataset else dataset["validation"]
    class_names = train_set.features["label"].names

    # 提取参数
    poison_rate = cfg.attack.poison_rate
    target_label = cfg.attack.target_label
    trigger_size = cfg.attack.trigger_size
    trigger_value = cfg.attack.trigger_value / 255.0
    attack_mode = getattr(cfg.attack, "attack_mode", "all2one") 
    num_classes = cfg.dataset.num_classes # 从 cfg 中获取类别数量

    base_transform, normalize_transform = _build_transform(cfg)

    # 2. 实例化全局 Transform 对象
    train_transform_obj = TrainTransform(
        base_transform, normalize_transform, poison_rate, target_label, trigger_size, trigger_value, attack_mode, num_classes
    )
    eval_clean_obj = EvalCleanTransform(base_transform, normalize_transform)
    eval_poison_obj = EvalPoisonTransform(
        base_transform, normalize_transform, target_label, trigger_size, trigger_value, attack_mode, num_classes
    )

    # 3. 绑定 Transform 并生成 DataLoader
    train_set.set_transform(train_transform_obj)
    clean_val_set = val_set.with_transform(eval_clean_obj)
    poison_val_set = val_set.with_transform(eval_poison_obj)

    # 创建 DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    clean_loader = DataLoader(
        clean_val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    poison_loader = DataLoader(
        poison_val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, clean_loader, poison_loader, class_names