# example_eval.py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# CIFAR-10 类别映射
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

def get_transforms_from_cfg(cfg: DictConfig):
    width = cfg.dataset.width
    height = cfg.dataset.height
    mean = cfg.dataset.mean
    std = cfg.dataset.std
    
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    IMAGE_PATH = "./docs/example_dog.png"
    CHECKPOINT_PATH = "./checkpoints/cifar10/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # 1. 验证文件路径
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"❌ 找不到图片文件: {IMAGE_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ 找不到权重文件: {CHECKPOINT_PATH}")

    # 2. 模型重建
    print(f"🏗️  Instantiating model: {cfg.model._target_}...")
    model = instantiate(cfg.model)

    # 3. 加载权重
    print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # 剥离外壳，仅加载模型权重
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint) 
        
    model = model.to(device)
    model.eval() # 切记：关闭 Dropout 和 BatchNorm 的训练模式

    # 4. 预处理对齐
    transform = get_transforms_from_cfg(cfg)
    try:
        image = Image.open(IMAGE_PATH).convert("RGB") 
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # 5. 维度升格: [C, H, W] -> [1, C, H, W]
    img_tensor = transform(image) 
    input_batch = img_tensor.unsqueeze(0).to(device) 

    # 6. 模型前向传播
    with torch.no_grad(): 
        outputs = model(input_batch)
        probabilities = F.softmax(outputs, dim=1)[0] 
        
        confidence_score, predicted_idx = torch.max(probabilities, dim=0)
        
        predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
        confidence_percent = confidence_score.item() * 100

    print("\n" + "=" * 40)
    print("📊 Prediction Results")
    print("=" * 40)
    print(f"🖼️  Image        : {IMAGE_PATH}")
    print(f"🎯 Target Class : {predicted_class}")
    print(f"🔥 Confidence   : {confidence_percent:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    main()