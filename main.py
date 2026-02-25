import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.data.pipeline import build_dataloaders, _visualize_batch
from src.engine.trainer import Trainer

from src.classifier_models import PreActResNet18, ResNet18, DenseNet121, MobileNetV2

def build_optimizer(cfg, model):
    """
    构造优化器和学习率调度器 (Scheduler), 从 config.yaml 里面读取, 沿用原代码中的内容
    """
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg.lr, 
        momentum=cfg.momentum, 
        weight_decay=cfg.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=cfg.scheduler_milestones, 
        gamma=cfg.scheduler_lambda
    )
    
    return optimizer, scheduler


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 全局随机种子固定 (包含 random, numpy, torch, cuda)
    set_seed(cfg.seed) 
    accelerator = Accelerator()
    train_loader, clean_loader, poison_loader, class_names = build_dataloaders(cfg)
    
    if accelerator.is_main_process:
        _visualize_batch(train_loader, cfg, class_names, save_path="sample_batch.png")

    model = instantiate(cfg.model)
    print(f"current dataset: {cfg.dataset.name}")
    print(f"current model: {model.__class__.__name__}")

    optimizer, scheduler = build_optimizer(cfg, model)

    model, optimizer, train_loader, clean_loader, poison_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, clean_loader, poison_loader, scheduler
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        clean_loader=clean_loader,   
        poison_loader=poison_loader, 
        accelerator=accelerator,
        cfg=cfg,
    )
    
    trainer.run()


if __name__ == "__main__":
    main()